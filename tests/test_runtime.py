"""The runtime scripts (runtime/train_ppo.py, runtime/eval_hover.py) on CPU without stable-baselines3, torch,
gymnasium, PyBullet or gym-pybullet-drones: stand-in modules replace them, so what runs is the scripts' own code (the
settings, the seeded initial states and episode log of `HoverItemEnv`, the rollout log, the cut writer, the per-item
scoring and the arm writer), and its outputs are checked with the adapter and gatedtrain's arm reader, as ext-collect
reads them.

Why: these scripts run only in the training and evaluation containers, so a fault in their pure parts was found only
after the run. The stand-ins model the documented behaviour the scripts rely on (runtime/README.md "Seeds" and "SB3
facts"): `VecEnv.seed(s)` makes env i reset first with seed s + i, automatic resets pass no seed, the callback's
`_on_rollout_start` / `_on_rollout_end` bracket each rollout of n_steps x n_envs env steps.
"""

from __future__ import annotations

import collections
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from gatedtrain.canonical import file_sha256
from gatedtrain.ext.arm import read_lines, read_meta
from gatedtrain_recipe_sb3_ppo_drone_hover.adapter import Sb3PpoAdapter

from _sb3 import EVALUATOR_ID, SAMPLER, SUPPORT_ID

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime"
SNAPSHOT = RUNTIME / "env-snapshot"
ITEMS = RUNTIME / "items"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"sb3_runtime_{name}", RUNTIME / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_ppo = load("train_ppo")
eval_hover = load("eval_hover")


# ---------------------------------------------------------------------------
# Stand-in modules
# ---------------------------------------------------------------------------

EPISODE_STEPS = 5          # stand-in HoverAviary truncates after this many steps, so episodes span rollouts


class FakeEnv:
    """gymnasium.Env: reset(seed) seeds np_random as gymnasium.utils.seeding.np_random does."""

    _np_random = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))

    @property
    def np_random(self):
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        return self._np_random


class FakeBaseAviary(FakeEnv):
    def reset(self, seed=None, options=None):
        self.pos = np.array([list(self.INIT_XYZS[0])], dtype=float)
        self.rpy = np.array(self.INIT_RPYS, dtype=float)
        self.t = 0

    def _updateAndStoreKinematicInformation(self):
        pass

    def _computeObs(self):
        return np.zeros((1, 27))

    def _computeInfo(self):
        return {}


class FakeHoverAviary(FakeBaseAviary):
    def __init__(self, *, drone_model, physics, pyb_freq, ctrl_freq, gui, record, obs, act):
        self.kwargs = dict(drone_model=drone_model, physics=physics, pyb_freq=pyb_freq, ctrl_freq=ctrl_freq,
                           gui=gui, record=record, obs=obs, act=act)
        self.ACTION_BUFFER_SIZE = ctrl_freq // 2
        self.action_buffer = collections.deque(maxlen=self.ACTION_BUFFER_SIZE)
        self.NUM_DRONES = 1
        self.action_space = types.SimpleNamespace(shape=(1, 1))
        self.DRONE_IDS = [7]
        self.CLIENT = 0

    def step(self, action):
        self.t += 1
        self.action_buffer.append(np.asarray(action).reshape(1, 1))
        return self._computeObs(), 1.0, False, self.t >= EPISODE_STEPS, {}


class FakeVecEnv:
    """DummyVecEnv: seed(s) sets env i's next reset seed to s + i, cleared after that reset; automatic resets pass no
    seed (stable_baselines3 base_vec_env.py:308, dummy_vec_env.py:71, :78-81)."""

    def __init__(self, fns, start_method=None):
        self.envs = [fn() for fn in fns]
        self.num_envs = len(self.envs)
        self._seeds = [None] * self.num_envs

    def seed(self, seed):
        self._seeds = [seed + i for i in range(self.num_envs)]

    def reset(self):
        for i, env in enumerate(self.envs):
            env.reset(seed=self._seeds[i])
        self._seeds = [None] * self.num_envs

    def step(self):
        for env in self.envs:
            _, _, terminated, truncated, _ = env.step(np.zeros((1, 1)))
            if terminated or truncated:
                env.reset()

    def env_method(self, name):
        return [getattr(env, name)() for env in self.envs]

    def close(self):
        self.closed = True


class FakeVecNormalize:
    def __init__(self, venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99):
        self.venv, self.norm_obs, self.norm_reward, self.clip_obs, self.gamma = venv, norm_obs, norm_reward, clip_obs, gamma
        self.clip_reward, self.epsilon, self.training = 10.0, 1e-8, True
        self.obs_rms = types.SimpleNamespace(mean=np.zeros(27), var=np.ones(27), count=1e-4)
        self.ret_rms = types.SimpleNamespace(mean=np.float64(0.0), var=np.float64(1.0), count=1e-4)

    def __getattr__(self, name):
        return getattr(self.venv, name)

    def normalize_obs(self, obs):
        return obs

    def save(self, path):
        Path(path).write_bytes(b"vecnormalize pickle stand-in " + str(self.obs_rms.count).encode())

    @classmethod
    def load(cls, path, venv):
        assert Path(path).is_file()
        return cls(venv)


class FakeBaseCallback:
    def __init__(self, verbose=0):
        self.model = None

    def init_callback(self, model):
        self.model = model
        self.training_env = model.env


class FakePPO:
    """PPO: learn() runs the callback's rollout hooks around rollouts of n_steps x n_envs env steps, as
    OnPolicyAlgorithm.learn / collect_rollouts do; save/load keep the settings and counters in model.zip."""

    def __init__(self, policy, env, *, n_steps, batch_size, n_epochs, seed, device, verbose=0, policy_kwargs=None,
                 **kw):
        self.env, self.n_steps, self.batch_size, self.n_epochs, self.seed = env, n_steps, batch_size, n_epochs, seed
        self.policy_kwargs = policy_kwargs or {}
        self.num_timesteps, self._n_updates = 0, 0
        self.policy = types.SimpleNamespace()
        self.observation_space = types.SimpleNamespace(shape=(1, 27))
        self.action_space = types.SimpleNamespace(shape=(1, 1))
        env.seed(seed)

    def save(self, path):
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("data", json.dumps({"n_steps": self.n_steps, "batch_size": self.batch_size,
                                           "n_epochs": self.n_epochs, "num_timesteps": self.num_timesteps,
                                           "n_updates": self._n_updates, "net_arch": self.policy_kwargs.get("net_arch")}))
            z.writestr("policy.pth", f"weights after {self._n_updates} updates".encode())

    @classmethod
    def load(cls, path, env=None, device="cpu", seed=None):
        with zipfile.ZipFile(path) as z:
            d = json.loads(z.read("data"))
        model = cls("MlpPolicy", env or FakeVecEnv([]), n_steps=d["n_steps"], batch_size=d["batch_size"], n_epochs=d["n_epochs"],
                    seed=seed, device=device, policy_kwargs={"net_arch": d["net_arch"]})
        model.num_timesteps, model._n_updates = d["num_timesteps"], d["n_updates"]
        return model

    def get_vec_normalize_env(self):
        return self.env

    def predict(self, obs, deterministic=False):
        assert deterministic is True
        return np.zeros((1, 1)), None

    def learn(self, total_timesteps, callback, reset_num_timesteps=True):
        if reset_num_timesteps:
            self.num_timesteps = 0
        else:
            total_timesteps += self.num_timesteps
        callback.init_callback(self)
        self.env.reset()
        while self.num_timesteps < total_timesteps:
            callback._on_rollout_start()
            for _ in range(self.n_steps):
                self.env.step()
                self.num_timesteps += self.env.num_envs
            callback._on_rollout_end()
            self._n_updates += self.n_epochs
        callback._on_training_end()
        return self


def stub_modules(root: Path) -> dict:
    """sys.modules entries for every heavy import of the two scripts."""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m

    pkg = root / "gym_pybullet_drones"
    (pkg / "assets").mkdir(parents=True, exist_ok=True)
    (pkg / "assets" / "cf2x.urdf").write_text("<robot/>\n", encoding="utf-8")
    velocities = []
    torch_threads = {"n": 1}
    return {
        "numpy": np,
        "gymnasium": mod("gymnasium", Env=FakeEnv),
        "pybullet": mod("pybullet", resetBaseVelocity=lambda *a, **kw: velocities.append((a, kw)), velocities=velocities),
        "gym_pybullet_drones": mod("gym_pybullet_drones", __file__=str(pkg / "__init__.py")),
        "gym_pybullet_drones.envs": mod("gym_pybullet_drones.envs"),
        "gym_pybullet_drones.envs.BaseAviary": mod("gym_pybullet_drones.envs.BaseAviary", BaseAviary=FakeBaseAviary),
        "gym_pybullet_drones.envs.HoverAviary": mod("gym_pybullet_drones.envs.HoverAviary", HoverAviary=FakeHoverAviary),
        "gym_pybullet_drones.utils": mod("gym_pybullet_drones.utils"),
        "gym_pybullet_drones.utils.enums": mod("gym_pybullet_drones.utils.enums", ActionType=str, DroneModel=str,
                                               ObservationType=str, Physics=str),
        "torch": mod("torch", set_num_threads=lambda n: torch_threads.update(n=n),
                     get_num_threads=lambda: torch_threads["n"]),
        "stable_baselines3": mod("stable_baselines3", PPO=FakePPO),
        "stable_baselines3.common": mod("stable_baselines3.common"),
        "stable_baselines3.common.vec_env": mod("stable_baselines3.common.vec_env", DummyVecEnv=FakeVecEnv,
                                                SubprocVecEnv=FakeVecEnv, VecNormalize=FakeVecNormalize),
        "stable_baselines3.common.callbacks": mod("stable_baselines3.common.callbacks", BaseCallback=FakeBaseCallback),
        "stable_baselines3.common.utils": mod("stable_baselines3.common.utils", set_random_seed=lambda s: None),
    }


_real_version = importlib.metadata.version
VERSIONS = {"gym-pybullet-drones": "2.2.0", "pybullet": "3.2.7", "stable-baselines3": "2.9.2"}


def fake_version(name):
    return VERSIONS[name] if name in VERSIONS else _real_version(name)


@contextlib.contextmanager
def stubbed(root: Path):
    with mock.patch.dict(sys.modules, stub_modules(root)), \
            mock.patch.object(importlib.metadata, "version", fake_version), \
            contextlib.redirect_stdout(io.StringIO()):
        yield


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# Settings, environment parameters and items: pinned to the README and the committed files
# ---------------------------------------------------------------------------

class TestPinnedSettings(unittest.TestCase):
    def test_ppo_settings_are_the_readme_profiles(self):
        # runtime/README.md "Profiles and expected cost" (decided 2026-11-12).
        self.assertEqual({k: train_ppo.COMMON[k] for k in ("n_envs", "n_steps", "batch_size", "n_epochs",
                                                            "learning_rate", "gamma", "gae_lambda", "clip_range",
                                                            "ent_coef", "vf_coef", "max_grad_norm", "net_arch",
                                                            "clip_obs", "torch_threads", "vec_env")},
                         {"n_envs": 8, "n_steps": 1024, "batch_size": 256, "n_epochs": 10, "learning_rate": 3.0e-4,
                          "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5,
                          "max_grad_norm": 0.5, "net_arch": [64, 64], "clip_obs": 10.0, "torch_threads": 4,
                          "vec_env": "subproc"})
        self.assertEqual(train_ppo.PHASE_SEED, {"base": 1000, "continue": 2000})

    def cuts(self, phase, profile, first=0):
        args = train_ppo.parse_args(["--phase", phase, "--profile", profile, "--out", "/o"]
                                    + (["--base", "/b"] if phase == "continue" else []))
        s = train_ppo.resolve_settings(args)
        return s, [first + i for i in range(1, s["iterations"] + 1) if i % s["save_every"] == 0]

    def test_cut_iterations_are_the_readme_table(self):
        s, cuts = self.cuts("base", "full")
        self.assertEqual((cuts, s["total_timesteps"], s["seed"]), ([4, 8], 65536, 1000))
        s, cuts = self.cuts("base", "smoke")
        self.assertEqual((cuts, s["total_timesteps"]), ([1, 2], 16384))
        s, cuts = self.cuts("continue", "smoke", first=8)
        self.assertEqual((cuts, s["total_timesteps"], s["seed"]), ([12, 16, 20], 98304, 2000))
        s, cuts = self.cuts("continue", "full", first=8)
        self.assertEqual((cuts, s["total_timesteps"]), ([18, 28, 38, 48, 58], 409600))

    def test_settings_refusals(self):
        def settings(*extra):
            return train_ppo.resolve_settings(train_ppo.parse_args(["--phase", "base", "--profile", "full", "--out",
                                                                    "/o", *extra]))
        with self.assertRaisesRegex(SystemExit, "multiple of batch_size"):
            settings("--n-steps", "100")
        with self.assertRaisesRegex(SystemExit, "last iteration is a cut"):
            settings("--iterations", "7")
        self.assertEqual(settings("--seed", "5", "--vec-env", "dummy")["seed"], 5)
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (["--phase", "continue", "--profile", "full", "--out", "/o"],
                         ["--phase", "base", "--profile", "full", "--out", "/o", "--base", "/b"],
                         ["--phase", "base", "--profile", "full"]):
                with self.assertRaises(SystemExit, msg=argv):
                    train_ppo.parse_args(argv)

    def test_committed_snapshot_is_the_scripts_env_params(self):
        params, shard = train_ppo.read_env_params(SNAPSHOT)
        self.assertEqual(params, train_ppo.ENV_PARAMS_V1)
        self.assertEqual(eval_hover.read_env_params(SNAPSHOT), params)
        self.assertEqual(params["env_params_version"], eval_hover.ENV_PARAMS_VERSION)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "shards").mkdir()
            bad = dict(params, decimals=3)
            (Path(tmp) / "shards" / shard.name).write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "is not the env_params_version"):
                train_ppo.read_env_params(Path(tmp))
            (Path(tmp) / "shards" / shard.name).write_text(json.dumps(params) + "\n" + json.dumps(params) + "\n")
            with self.assertRaisesRegex(SystemExit, "exactly one line"):
                train_ppo.read_env_params(Path(tmp))

    def test_env_class_is_identical_in_both_scripts_and_the_snapshot(self):
        # eval_hover.py refuses a cut whose env_version.json names another HoverItemEnv hash.
        self.assertEqual(train_ppo.env_class_sha256(), eval_hover.env_class_sha256())
        recorded = json.loads((SNAPSHOT / "env_version.json").read_text(encoding="utf-8"))["env_class_sha256"]
        self.assertEqual(train_ppo.env_class_sha256(), recorded)
        self.assertEqual(json.loads((SNAPSHOT / "env_version.json").read_text())["env_kwargs"], train_ppo.ENV_KWARGS)

    def test_payload_text_is_shared_and_matches_the_items(self):
        rows = read_jsonl(ITEMS / "items.jsonl")
        self.assertEqual(len(rows), 300)
        for r in rows:
            self.assertEqual(train_ppo.init_payload(r["env_params_version"], r["init"]), r["payload"])
            self.assertEqual(eval_hover.init_payload(r["env_params_version"], r["init"]), r["payload"])
        self.assertEqual(eval_hover.read_items(ITEMS), rows)

    def test_make_items_reproduces_the_registered_item_set(self):
        meta = json.loads((ITEMS / "item_set_meta.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "items"
            with contextlib.redirect_stdout(io.StringIO()):
                eval_hover.make_items(meta["n"], meta["item_seed"], SNAPSHOT, out)
            self.assertEqual((out / "items.jsonl").read_bytes(), (ITEMS / "items.jsonl").read_bytes())
            got = json.loads((out / "item_set_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(got["items_sha256"], meta["items_sha256"])
            self.assertEqual(got["env_params_sha256"], meta["env_params_sha256"])
            with self.assertRaisesRegex(SystemExit, "not empty"):
                eval_hover.make_items(3, 1, SNAPSHOT, out)

    def test_read_items_refuses_a_payload_that_is_not_its_init(self):
        rows = read_jsonl(ITEMS / "items.jsonl")[:3]
        with tempfile.TemporaryDirectory() as tmp:
            rows[1] = dict(rows[1], init=dict(rows[1]["init"], z=rows[1]["init"]["z"] + 0.0001))
            (Path(tmp) / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(SystemExit, "payload does not match"):
                eval_hover.read_items(Path(tmp))
            rows[1] = rows[0]
            (Path(tmp) / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(SystemExit, "not unique"):
                eval_hover.read_items(Path(tmp))


# ---------------------------------------------------------------------------
# HoverItemEnv (the launcher-side environment subclass)
# ---------------------------------------------------------------------------

class TestHoverItemEnv(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ctx = stubbed(Path(tmp.name))
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.cls = train_ppo.make_hover_item_env_class()

    def env(self, rank=0):
        return self.cls(train_ppo.ENV_PARAMS_V1, train_ppo.ENV_KWARGS, rank)

    def test_seeded_initial_states_are_one_stream_per_env(self):
        a, b = self.env(), self.env()
        a.reset(seed=1003)
        b.reset(seed=1003)
        for _ in range(3):
            a.reset()
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(1003)))
        want = [{k: round(float(rng.uniform(*train_ppo.ENV_PARAMS_V1["init_ranges"][k])), 4)
                 for k in ("x", "y", "z", "vz")} for _ in range(4)]
        self.assertEqual([e["init"] for e in a._log], want)
        self.assertEqual(b._log[0]["init"], want[0])
        self.assertEqual({e["seed"] for e in a._log}, {1003})
        self.assertEqual([e["episode"] for e in a._log], [0, 1, 2, 3])
        lo = {k: v[0] for k, v in train_ppo.ENV_PARAMS_V1["init_ranges"].items()}
        hi = {k: v[1] for k, v in train_ppo.ENV_PARAMS_V1["init_ranges"].items()}
        for e in a._log:
            self.assertTrue(all(lo[k] <= e["init"][k] <= hi[k] for k in lo), e)

    def test_reset_applies_the_initial_state_and_clears_the_action_buffer(self):
        env = self.env()
        env.reset(seed=1)
        env.step(np.ones((1, 1)))
        self.assertTrue(np.any(np.array(env.action_buffer)))
        init = {"x": 0.1, "y": -0.2, "z": 1.5, "vz": 0.3}
        env.reset(options={"init": init})
        self.assertEqual(len(env.action_buffer), env.ACTION_BUFFER_SIZE)
        self.assertFalse(np.any(np.array(env.action_buffer)))
        self.assertEqual(env.INIT_XYZS.tolist(), [[0.1, -0.2, 1.5]])
        self.assertEqual(env.INIT_RPYS.tolist(), [[0.0, 0.0, 0.0]])
        velocity = sys.modules["pybullet"].velocities[-1][0]
        self.assertEqual((velocity[0], velocity[1]), (7, [0.0, 0.0, 0.3]))
        self.assertEqual(env._log[-1]["init"], init)

    def test_rollout_log_counts_every_step_once(self):
        env = self.env()
        env.reset(seed=3)
        logs = []
        for chunk in (3, 4, 6):                 # rollouts that end mid-episode
            for _ in range(chunk):
                if env.step(np.zeros((1, 1)))[3]:
                    env.reset()
            logs.append(env.pop_rollout_log())
        self.assertEqual([sum(e["steps"] for e in log) for log in logs], [3, 4, 6])
        # The episode open at a rollout end continues in the next rollout's log, with its own steps there.
        self.assertEqual(logs[0][-1]["episode"], logs[1][0]["episode"])
        self.assertEqual([e["steps"] for e in logs[1]], [2, 2])


# ---------------------------------------------------------------------------
# train_ppo.py end to end with the stand-ins; outputs read by the adapter
# ---------------------------------------------------------------------------

class TestTrainOutputs(unittest.TestCase):
    """A base run (2 iterations, cuts 1 and 2) and a continuation from cut 2 (smoke: 12 iterations, cuts 6, 10, 14),
    2 envs x 8 steps per rollout; then the adapter's cut, consumption, overlap and binding checks on what they wrote."""

    SMALL = ["--n-envs", "2", "--n-steps", "8", "--batch-size", "8", "--vec-env", "dummy"]

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.snap, cls.base_out, cls.cont_out = root / "snap", root / "base", root / "cont"
        with stubbed(root / "stubs"):
            train_ppo.main(["--write-env-snapshot", str(cls.snap)])
            train_ppo.main(["--phase", "base", "--profile", "smoke", "--out", str(cls.base_out),
                            "--env-snapshot", str(cls.snap), *cls.SMALL])
            cls.base_cut = cls.base_out / "checkpoint-2"
            train_ppo.main(["--phase", "continue", "--profile", "smoke", "--out", str(cls.cont_out),
                            "--env-snapshot", str(cls.snap), "--base", str(cls.base_cut), *cls.SMALL])
        cls.adapter = Sb3PpoAdapter()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_snapshot_is_one_shard_with_the_params(self):
        self.assertEqual(read_jsonl(self.snap / "shards" / "hover-init-v1.jsonl"), [train_ppo.ENV_PARAMS_V1])
        doc = json.loads((self.snap / "env_version.json").read_text())
        self.assertEqual(doc["env_class_sha256"], eval_hover.env_class_sha256())

    def test_cuts_are_complete_for_the_adapter(self):
        for out, want in ((self.base_out, [1, 2]), (self.cont_out, [6, 10, 14])):
            complete, incomplete = self.adapter.find_cuts(out)
            self.assertEqual(([s for s, _ in complete], incomplete), (want, []))
            self.assertEqual(list(out.glob(".checkpoint-*")), [])
        marker = json.loads((self.cont_out / "checkpoint-14" / "cut.json").read_text())
        self.assertEqual((marker["iteration"], marker["num_timesteps"], marker["phase"]), (14, 14 * 16, "continue"))
        self.assertEqual(json.loads((self.base_cut / "env_version.json").read_text()),
                         json.loads((self.snap / "env_version.json").read_text()))

    def test_consumption_log_passes_the_adapter(self):
        data = self.snap / "shards"
        for out in (self.base_out, self.cont_out):
            cuts, _ = self.adapter.find_cuts(out)
            self.assertEqual(self.adapter.consumption_problems(out, data, cuts), [])
        cuts, _ = self.adapter.find_cuts(self.cont_out)
        rows = self.adapter.consumption(self.cont_out, data, cuts, run_id="r")["rows"]
        self.assertEqual([r["step"] for r in rows], list(range(3, 15)))

    def test_rollout_lines(self):
        lines = read_jsonl(self.cont_out / "rollouts.jsonl")
        self.assertEqual([d["iteration"] for d in lines], list(range(3, 15)))
        self.assertEqual((lines[0]["num_timesteps_first"], lines[-1]["num_timesteps_last"]), (33, 224))
        for d in lines:
            for e in d["episodes"]:
                self.assertEqual(e["payload"], eval_hover.init_payload("hover-init-v1", e["init"]))
        # Env i's episodes carry the run's seed + i: the continuation draws other states than the base run.
        first = {e["env"]: e["seed"] for e in lines[0]["episodes"]}
        self.assertEqual(first, {0: 2000, 1: 2001})
        base_first = {e["env"]: e["seed"] for e in read_jsonl(self.base_out / "rollouts.jsonl")[0]["episodes"]}
        self.assertEqual(base_first, {0: 1000, 1: 1001})

    def test_evaluation_worker_loads_the_cut_frozen_and_applies_the_item_state(self):
        item = read_jsonl(ITEMS / "items.jsonl")[0]
        with stubbed(Path(self._tmp.name) / "stubs-eval"), mock.patch.dict(eval_hover._WORKER):
            eval_hover._worker_init(str(self.cont_out / "checkpoint-14"), train_ppo.ENV_PARAMS_V1,
                                    train_ppo.ENV_KWARGS, 1, 0, "hover_success")
            vn, env = eval_hover._WORKER["vn"], eval_hover._WORKER["env"]
            self.assertEqual((vn.training, vn.norm_reward), (False, False))
            r = eval_hover.run_item(item)
        self.assertIsNone(r["error"])
        self.assertEqual(env._log[-1]["init"], item["init"])
        self.assertEqual(env.INIT_XYZS.tolist(), [[item["init"][k] for k in ("x", "y", "z")]])
        self.assertEqual((r["sub_metrics"]["steps"], r["sub_metrics"]["crashed"], r["score"]), (EPISODE_STEPS, 1, 0.0))

    def test_training_payloads_are_disjoint_from_the_items(self):
        cuts, _ = self.adapter.find_cuts(self.cont_out)
        payloads, why = self.adapter.training_payloads(self.cont_out, self.base_cut, cuts)
        self.assertIsNone(why)
        n_episodes = sum(len(d["episodes"]) for out in (self.base_out, self.cont_out)
                         for d in read_jsonl(out / "rollouts.jsonl"))
        self.assertGreater(len(payloads), 20)
        self.assertLessEqual(len(payloads), n_episodes)
        self.assertFalse(payloads & {r["payload"] for r in read_jsonl(ITEMS / "items.jsonl")})

    def test_binding_checks_pass(self):
        mounts = [{"host": str(self.base_cut), "container": str(self.base_cut), "mode": "ro"},
                  {"host": str(self.snap), "container": str(self.snap), "mode": "ro"}]
        launch = {"mounts": mounts, "training_seed": 2000, "profile": "smoke"}
        env = file_sha256(self.snap / "env_version.json")
        problems = self.adapter.checks(last_cut=self.cont_out / "checkpoint-14", base=self.base_cut, launch=launch,
                                       base_field="base", data=self.snap / "shards",
                                       eval_distribution={"env_version": env, "env_params_version": "hover-init-v1"})
        self.assertEqual(problems, [])

    def test_run_records(self):
        args = json.loads((self.cont_out / "run_args.json").read_text())
        self.assertEqual((args["first_iteration"], args["last_iteration"], args["seed"]), (3, 14, 2000))
        self.assertEqual(args["env_seeds"], [2000, 2001])
        self.assertEqual(args["settings"]["rollout_size"], 16)
        attempts = read_jsonl(self.cont_out / "attempts.jsonl")
        self.assertEqual([a.get("last_iteration") for a in attempts], [None, 14])
        self.assertEqual(len(read_jsonl(self.cont_out / "memory.jsonl")), 12)
        freeze = (self.cont_out / "env-freeze.txt").read_text()
        self.assertEqual(json.loads((self.cont_out / "env-runtime.json").read_text())["env_freeze_sha256"],
                         file_sha256(self.cont_out / "env-freeze.txt"))
        self.assertIn("numpy==", freeze)


class TestTrainHelpers(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_prepare_out_moves_a_preempted_attempt_aside(self):
        out = self.tmp / "out"
        self.assertEqual(train_ppo.prepare_out(out), 1)
        (out / "rollouts.jsonl").write_text("{}\n")
        self.assertEqual(train_ppo.prepare_out(out), 2)
        self.assertTrue((out / "preempted-attempt-1" / "rollouts.jsonl").is_file())
        self.assertFalse((out / "rollouts.jsonl").exists())
        self.assertEqual([a["attempt"] for a in read_jsonl(out / "attempts.jsonl")], [1, 2])

    def test_read_base_refuses_an_incomplete_or_altered_cut(self):
        cut = self.tmp / "checkpoint-2"
        cut.mkdir()
        with self.assertRaisesRegex(SystemExit, "not a complete cut"):
            train_ppo.read_base(cut)
        (cut / "policy.pth").write_bytes(b"w")
        (cut / "cut.json").write_text(json.dumps({"iteration": 2, "files": {"policy.pth": file_sha256(cut / "policy.pth")}}))
        self.assertEqual(train_ppo.read_base(cut)["iteration"], 2)
        (cut / "policy.pth").write_bytes(b"x")
        with self.assertRaisesRegex(SystemExit, "differs from its SHA-256"):
            train_ppo.read_base(cut)


# ---------------------------------------------------------------------------
# eval_hover.py: per-item scoring and the arm
# ---------------------------------------------------------------------------

class ScriptedEnv:
    """A finished episode as a list of (z, roll, reward, terminated, truncated) per control step."""

    def __init__(self, script):
        self.script = script

    def reset(self, seed=None, options=None):
        self.k = 0
        self.pos, self.rpy = np.array([[0.0, 0.0, 0.5]]), np.zeros((1, 3))
        return np.zeros((1, 27)), {}

    def step(self, action):
        z, roll, reward, terminated, truncated = self.script[self.k]
        self.k += 1
        self.pos, self.rpy = np.array([[0.2, 0.0, z]]), np.array([[roll, 0.0, 0.0]])
        return np.zeros((1, 27)), reward, terminated, truncated, {}


class FakePolicy:
    def predict(self, obs, deterministic):
        assert deterministic is True
        return np.zeros((1, 1)), None


def episode(n, z=1.0, *, end="time", roll=0.0, reward=2.0):
    """n steps at height z; the last step truncates by time (n = 242), by a crash, or terminates on the target."""
    steps = [(z, roll, reward, False, False) for _ in range(n)]
    steps[-1] = (z, roll, reward, end == "target", end in ("time", "crash"))
    return steps


class TestRunItem(unittest.TestCase):
    """`hover_success`: fraction of steps 121 .. 242 inside the bounds with |z - 1| <= 0.06 (runtime/README.md)."""

    def score(self, script, metric="hover_success"):
        identity = types.SimpleNamespace(normalize_obs=lambda o: o)
        with mock.patch.dict(eval_hover._WORKER, env=ScriptedEnv(script), vn=identity, model=FakePolicy(), metric=metric):
            return eval_hover.run_item({"item": "i", "init": {}})

    def test_full_hover(self):
        r = self.score(episode(242))
        self.assertEqual((r["score"], r["error"]), (1.0, None))
        self.assertEqual(r["sub_metrics"]["hover_score"], 1.0)
        self.assertEqual((r["sub_metrics"]["steps"], r["sub_metrics"]["crashed"]), (242, 0))
        self.assertEqual(r["sub_metrics"]["success_3d_frac"], 0.0)        # x offset 0.2 > 0.06
        self.assertEqual(r["sub_metrics"]["within_10cm_frac"], 0.0)        # d >= 0.2
        self.assertEqual(self.score(episode(242), metric="hover_score")["score"], 1.0)

    def test_only_the_second_half_counts(self):
        script = [(0.5, 0.0, 1.0, False, False)] * 120 + episode(122)
        self.assertEqual(self.score(script)["score"], 1.0)
        script = [(1.0, 0.0, 2.0, False, False)] * 121 + episode(121, z=0.5)
        self.assertAlmostEqual(self.score(script)["score"], 1 / 122)

    def test_tolerance_edges(self):
        self.assertEqual(self.score(episode(242, z=1.0599))["score"], 1.0)
        self.assertEqual(self.score(episode(242, z=1.0601))["score"], 0.0)
        self.assertEqual(self.score(episode(242, z=1.0, roll=0.41))["score"], 0.0)   # outside the tilt bound

    def test_crash_counts_zero_for_the_rest(self):
        r = self.score(episode(150, end="crash"))
        self.assertAlmostEqual(r["score"], 30 / 122)                      # steps 121 .. 150
        self.assertEqual((r["sub_metrics"]["crashed"], r["sub_metrics"]["steps"]), (1, 150))
        self.assertAlmostEqual(r["sub_metrics"]["hover_score"], 150 * 2 / 484)

    def test_reaching_the_target_credits_the_rest(self):
        for n in (100, 130):
            r = self.score(episode(n, end="target"))
            self.assertEqual(r["score"], 1.0, n)
            self.assertEqual(r["sub_metrics"]["hover_score"], 1.0)
            self.assertEqual(r["sub_metrics"]["crashed"], 0)

    def test_simulator_exception_is_an_error_record(self):
        r = self.score(episode(10)[:-1] + [(1.0, 0.0, 2.0, False, False)])   # runs off the script
        self.assertIsNone(r["score"])
        self.assertEqual(r["error"]["class"], "simulator_error")
        self.assertIsNone(r["sub_metrics"])


class TestEvaluateArm(unittest.TestCase):
    """evaluate() on a cut laid out as save_cut writes it, with stand-in policy and env; the arm is read with
    gatedtrain's arm reader against the adapter's evaluator rows."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        cut = self.tmp / "checkpoint-58"
        cut.mkdir()
        (cut / "policy.pth").write_bytes(b"weights")
        with zipfile.ZipFile(cut / "model.zip", "w") as z:
            z.writestr("policy.pth", b"weights")
        for name in ("policy_meta.json", "vecnormalize.json"):
            (cut / name).write_text("{}\n")
        (cut / "vecnormalize.pkl").write_bytes(b"pickle stand-in")
        shutil.copyfile(SNAPSHOT / "env_version.json", cut / "env_version.json")
        files = {n: file_sha256(cut / n) for n in train_ppo.CUT_FILES}
        (cut / "cut.json").write_text(json.dumps({"iteration": 58, "num_timesteps": 475136, "files": files}))
        self.cut = cut
        items = self.tmp / "items"
        items.mkdir()
        every = read_jsonl(ITEMS / "items.jsonl")
        rows = [r for r in every if r["init"]["z"] > 1][:2] + [r for r in every if r["init"]["z"] <= 1][:2]
        rows = [rows[0], rows[2], rows[1], rows[3]]           # scores 1, 0, 1, 0
        (items / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.items, self.rows = items, rows

    def run_eval(self, metric, *, fail_item=None):
        def worker_init(cut, params, env_kwargs, threads, eval_seed, m):
            self.assertEqual(env_kwargs, train_ppo.ENV_KWARGS)
            scripts = collections.deque()
            for r in self.rows:
                z = r["init"]["z"]
                scripts.append([(z, 0.0, 1.0, False, False)] * 5 if r["item"] == fail_item else
                               episode(242, z=1.0 if z > 1 else 0.5, reward=1.5))

            class Env(ScriptedEnv):
                def reset(env, seed=None, options=None):
                    env.script = scripts.popleft()
                    return ScriptedEnv.reset(env)
            eval_hover._WORKER.update(env=Env(None), vn=types.SimpleNamespace(normalize_obs=lambda o: o),
                                      model=FakePolicy(), metric=m)

        arm = self.tmp / f"arm-{metric}"
        argv = ["--checkpoint", str(self.cut), "--items", str(self.items), "--arm-out", str(arm), "--workers", "1",
                "--env-snapshot", str(SNAPSHOT), "--metric", metric, "--model-hash", "sha256:" + "a" * 64,
                "--item-set-hash", "sha256:" + "b" * 64]
        with mock.patch.object(eval_hover, "_worker_init", worker_init), \
                mock.patch.object(importlib.metadata, "version", fake_version), \
                contextlib.redirect_stdout(io.StringIO()):
            eval_hover.main(argv)
        return arm

    def check_arm(self, arm, evaluator_id):
        adapter = Sb3PpoAdapter()
        row = adapter.metric(evaluator_id)
        meta, bad = read_meta(arm)
        self.assertEqual(bad, [])
        self.assertEqual((meta["evaluator_id"], meta["metric"], meta["direction"], meta["score_range"]),
                         (evaluator_id, row["metric"], row["direction"], list(row["score_range"])))
        self.assertEqual(meta["scorer"]["version"], row["scorer"]["version"])
        self.assertEqual(meta["sampler"], SAMPLER)
        self.assertEqual(meta["normalizer_sha256"], adapter.normalizer_sha256(self.cut))
        self.assertEqual(meta["env_version_sha256"], adapter.env_version_sha256(self.cut))
        records, bad = read_lines(arm, seeds=meta["eval_seeds"], score_range=row["score_range"],
                                  error_classes=row["error_classes"])
        self.assertEqual(bad, [])
        return meta, records

    def test_deciding_arm(self):
        arm = self.run_eval("hover_success")
        meta, records = self.check_arm(arm, EVALUATOR_ID)
        self.assertEqual([r["item"] for r in records], [r["item"] for r in self.rows])
        want = [1.0 if r["init"]["z"] > 1 else 0.0 for r in self.rows]
        self.assertEqual([r["score"] for r in records], want)
        self.assertEqual(want, [1.0, 0.0, 1.0, 0.0])
        self.assertEqual([r["item_meta"]["init"] for r in records], [r["init"] for r in self.rows])
        summary = json.loads((arm / "summary.json").read_text())
        self.assertEqual((summary["iteration"], summary["missing_hashes"]), (58, []))

    def test_supporting_arm_and_an_error_item(self):
        arm = self.run_eval("hover_score", fail_item=self.rows[2]["item"])
        meta, records = self.check_arm(arm, SUPPORT_ID)
        self.assertEqual([r["score"] for r in records][:2], [0.75, 0.75])
        self.assertEqual(records[2]["error"]["class"], "simulator_error")

    def test_refuses_a_cut_from_another_environment(self):
        doc = json.loads((self.cut / "env_version.json").read_text())
        doc["env_class_sha256"] = "sha256:" + "0" * 64
        (self.cut / "env_version.json").write_text(json.dumps(doc))
        marker = json.loads((self.cut / "cut.json").read_text())
        marker["files"]["env_version.json"] = file_sha256(self.cut / "env_version.json")
        (self.cut / "cut.json").write_text(json.dumps(marker))
        with self.assertRaisesRegex(SystemExit, "another HoverItemEnv"):
            eval_hover.check_cut(self.cut)

    def test_refuses_a_zip_whose_weights_are_not_the_cut(self):
        with zipfile.ZipFile(self.cut / "model.zip", "w") as z:
            z.writestr("policy.pth", b"other weights")
        marker = json.loads((self.cut / "cut.json").read_text())
        marker["files"]["model.zip"] = file_sha256(self.cut / "model.zip")
        (self.cut / "cut.json").write_text(json.dumps(marker))
        with self.assertRaisesRegex(SystemExit, "evaluated weights would differ"):
            eval_hover.check_cut(self.cut)


if __name__ == "__main__":
    unittest.main()
