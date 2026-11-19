#!/usr/bin/env python
"""SB3 PPO on gym-pybullet-drones hover for multimodal prototype area "control" (prototype, unpromotable).

Two phases, two runs:
  --phase base      train from scratch for a deliberately small budget: the under-trained base policy
  --phase continue  load a base cut (--base DIR) and train further: the candidate

Plain stable-baselines3 2.9.2 PPO on CPU (README.md "CPU, not GPU"). No framework source is edited; the
launcher-side additions are the environment subclass `HoverItemEnv` (seeded initial-state randomization, action
buffer cleared at reset, per-episode log), a callback that saves cuts at rollout boundaries, and a callback that
writes the observed per-rollout consumption log. Files are written at the top of --out, never inside a cut
after its `cut.json`.

Outputs in --out (the run's only read-write mount):
  env-freeze.txt       `name==version` lines of the runtime, written before torch is imported (MM-33)
  env-runtime.json     interpreter, image CUDA version, lock hash, script hash (EXT-83), env class hash
  run_args.json        every PPO / VecNormalize / env value as JSON, seeds and their derivation, the env-params
                       shard path and SHA-256 (data binding), the base cut and its hashes (continue phase)
  rollouts.jsonl       observed consumption log (MM-25, MM-27): one line per PPO iteration (rollout), written at
                       the rollout's end, before the update that consumes it
  memory.jsonl         per-iteration wall time and peak RSS
  attempts.jsonl       one line per start and finish of this command (gpu-step.sh reruns after a preemption)
  checkpoint-<I>/      cuts after PPO iteration I (global count, continuing across phases); `cut.json` last

Usage (inside the container; README.md has the docker command):
  python train_ppo.py --phase base --profile full --out /out --env-snapshot DIR
  python train_ppo.py --phase continue --profile smoke --out /out --env-snapshot DIR --base /inputs/base
  python train_ppo.py --write-env-snapshot DIR       # once, host side: materialize the env DatasetSnapshot
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import resource
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULT_SNAPSHOT = str(HERE / "env-snapshot")
SHARD_DIR = "shards"
ENV_VERSION_FILE = "env_version.json"
# The one environment-parameters version of this prototype (C3: one shard per env_params_version).
ENV_PARAMS_VERSION = "hover-init-v1"
ENV_PARAMS_V1 = {
    "env_params_version": ENV_PARAMS_VERSION,
    # Initial state drawn per episode, uniform, rounded to `decimals` (the rounded value is what the env applies).
    "init_ranges": {"x": [-0.25, 0.25], "y": [-0.25, 0.25], "z": [0.15, 1.85], "vz": [-0.5, 0.5]},
    "init_order": ["x", "y", "z", "vz"],
    "decimals": 4,
    "init_rpy": [0.0, 0.0, 0.0],
    "action_buffer_reset": "zeros at every reset",
}
# HoverAviary arguments (gym_pybullet_drones/envs/HoverAviary.py:11-22): one-dimensional RPM action (the
# example's default, examples/learn.py:43), kinematic observation, PyBullet physics at 240 Hz, control at 30 Hz.
ENV_KWARGS = {"drone_model": "cf2x", "physics": "pyb", "pyb_freq": 240, "ctrl_freq": 30, "obs": "kin",
              "act": "one_d_rpm", "gui": False, "record": False}

# PPO and VecNormalize settings shared by both phases (every value goes to run_args.json).
COMMON = {
    "n_envs": 8,
    "n_steps": 1024,                # per env per rollout: 8 x 1024 = 8192 env steps per PPO iteration
    "batch_size": 256,
    "n_epochs": 10,
    "learning_rate": 3.0e-4,        # constant
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch": [64, 64],           # SB3 MlpPolicy default for PPO, separate pi and vf networks
    "norm_obs": True,
    "norm_reward": True,
    "clip_obs": 10.0,
    "torch_threads": 4,
    "vec_env": "subproc",           # SubprocVecEnv, start method forkserver
}
PHASE_SEED = {"base": 1000, "continue": 2000}   # PPO seed; env i first resets with seed + i (README.md "Seeds")
PROFILES = {
    # (phase, profile): PPO iterations to run in this phase and the cut interval, in iterations.
    # Base (chosen 2026-11-12): cut 8 (65,536 env steps). Cut 4 never leaves the ground,
    # so the deciding T3 metric is 0 on every item there (README.md "Base policy").
    ("base", "smoke"): {"iterations": 2, "save_every": 1},
    ("base", "full"): {"iterations": 8, "save_every": 4},
    # Continue (chosen 2026-11-12): 2.1-2.7 s per iteration on a 32-core CPU; the score
    # saturates within about 10 iterations, so smoke is 12 iterations (about 40 s) and full 50 (about 2-3 min).
    ("continue", "smoke"): {"iterations": 12, "save_every": 4},
    ("continue", "full"): {"iterations": 50, "save_every": 10},
}
CUT_FILES = ("model.zip", "policy.pth", "policy_meta.json", "vecnormalize.pkl", "vecnormalize.json",
             ENV_VERSION_FILE)
CUT_MARKER = "cut.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def write_json(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def init_payload(version: str, init: dict) -> str:
    """Canonical text of one initial state; the exact-overlap check compares these strings (README.md)."""
    return f"{version} x={init['x']:.4f} y={init['y']:.4f} z={init['z']:.4f} vz={init['vz']:.4f}"


# ---------------------------------------------------------------------------
# Pre-step (MM-33): runtime freeze, written before torch is imported
# ---------------------------------------------------------------------------

def write_env_freeze(out: Path) -> None:
    pkgs = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pkgs[name.lower().replace("_", "-")] = dist.version
    lines = [f"{k}=={pkgs[k]}" for k in sorted(pkgs)]
    # gym-pybullet-drones comes from a git commit; PEP 610 direct_url.json names it.
    try:
        direct = importlib.metadata.distribution("gym-pybullet-drones").read_text("direct_url.json")
        commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    except importlib.metadata.PackageNotFoundError:
        commit = None
    if commit:
        lines.append(f"# gym-pybullet-drones commit {commit}")
    (out / "env-freeze.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    lock = HERE / "uv.lock"
    runtime = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "image_cuda_version": os.environ.get("CUDA_VERSION"),
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "env_class_sha256": env_class_sha256(),
        "gym_pybullet_drones_commit": commit,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "cpu_count": os.cpu_count(),
        "env_freeze_sha256": sha256_file(out / "env-freeze.txt"),
    }
    write_json(out / "env-runtime.json", runtime)


def prepare_out(out: Path) -> int:
    """Create --out; a non-empty --out is a rerun of the same command by gpu-step.sh after a preemption.
    Earlier contents move to `preempted-attempt-<k>/` (never deleted). Returns the attempt number."""
    out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out.iterdir() if not p.name.startswith("preempted-attempt-")]
    previous = sorted(p for p in out.iterdir() if p.name.startswith("preempted-attempt-"))
    attempt = len(previous) + 1
    if existing:
        attempt = len(previous) + 2
        dest = out / f"preempted-attempt-{attempt - 1}"
        dest.mkdir()
        for p in existing:
            p.rename(dest / p.name)
        if (dest / "attempts.jsonl").exists():
            shutil.copyfile(dest / "attempts.jsonl", out / "attempts.jsonl")
    append_jsonl(out / "attempts.jsonl", {"attempt": attempt, "started_at": utc_now(), "argv": sys.argv})
    return attempt


# ---------------------------------------------------------------------------
# Environment (launcher side; eval_hover.py carries a byte-identical copy of the class, checked by hash)
# ---------------------------------------------------------------------------

# BEGIN HoverItemEnv
def make_hover_item_env_class():
    import numpy as np
    import pybullet as p
    from gymnasium import Env
    from gym_pybullet_drones.envs.BaseAviary import BaseAviary
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics

    class HoverItemEnv(HoverAviary):
        """HoverAviary with a seeded initial state per episode, an action buffer cleared at every reset, and a
        per-episode log. Stock HoverAviary starts every episode at the same pose (BaseAviary.py:193-205) and
        keeps the previous episode's actions in its observation buffer (BaseRLAviary.py:66-67, :152-153)."""

        def __init__(self, env_params: dict, env_kwargs: dict, rank: int = 0):
            self.env_params = env_params
            self.rank = rank
            self.episode = -1
            self.episode_seed = None
            self._log: list[dict] = []
            super().__init__(drone_model=DroneModel(env_kwargs["drone_model"]), physics=Physics(env_kwargs["physics"]),
                             pyb_freq=env_kwargs["pyb_freq"], ctrl_freq=env_kwargs["ctrl_freq"],
                             gui=env_kwargs["gui"], record=env_kwargs["record"],
                             obs=ObservationType(env_kwargs["obs"]), act=ActionType(env_kwargs["act"]))

        def sample_init(self, rng) -> dict:
            prm = self.env_params
            return {k: round(float(rng.uniform(*prm["init_ranges"][k])), prm["decimals"]) for k in prm["init_order"]}

        def reset(self, seed=None, options=None):
            Env.reset(self, seed=seed)                      # seeds self.np_random when seed is given
            if seed is not None:
                self.episode_seed = seed
            init = dict(options["init"]) if options and "init" in options else self.sample_init(self.np_random)
            self.INIT_XYZS = np.array([[init["x"], init["y"], init["z"]]])
            self.INIT_RPYS = np.array([self.env_params["init_rpy"]], dtype=float)
            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES, self.action_space.shape[-1])))
            BaseAviary.reset(self, seed=None, options=None)   # resetSimulation, loads the drone at INIT_XYZS
            p.resetBaseVelocity(self.DRONE_IDS[0], [0.0, 0.0, init["vz"]], [0.0, 0.0, 0.0],
                                physicsClientId=self.CLIENT)
            self._updateAndStoreKinematicInformation()
            self.episode += 1
            self._log.append({"env": self.rank, "episode": self.episode, "seed": self.episode_seed,
                              "init": init, "steps": 0})
            return self._computeObs(), self._computeInfo()

        def step(self, action):
            out = super().step(action)
            self._log[-1]["steps"] += 1
            return out

        def pop_rollout_log(self) -> list[dict]:
            """Episodes with at least one step since the last call; the current episode stays open."""
            done = [dict(e) for e in self._log if e["steps"] > 0]
            current = self._log[-1]
            self._log = [{**current, "steps": 0}]
            return done

    return HoverItemEnv
# END HoverItemEnv


def env_class_sha256() -> str:
    """SHA-256 of the HoverItemEnv source block; eval_hover.py refuses a checkpoint trained with another one."""
    text = Path(__file__).resolve().read_text(encoding="utf-8")
    start, end = text.index("# BEGIN HoverItemEnv\n"), text.index("# END HoverItemEnv\n")
    return sha256_bytes(text[start:end].encode("utf-8"))


def env_version_doc() -> dict:
    """MM-19 `env_version.json`: package, commit, env id, kwargs, assets hash, launcher env class hash."""
    import gym_pybullet_drones

    assets = Path(gym_pybullet_drones.__file__).resolve().parent / "assets"
    h = hashlib.sha256()
    for f in sorted(p for p in assets.rglob("*") if p.is_file()):
        h.update(f.relative_to(assets).as_posix().encode() + b"\0" + sha256_file(f).encode() + b"\n")
    try:
        direct = importlib.metadata.distribution("gym-pybullet-drones").read_text("direct_url.json")
        commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    except importlib.metadata.PackageNotFoundError:
        commit = None
    return {
        "package": "gym-pybullet-drones",
        "version": importlib.metadata.version("gym-pybullet-drones"),
        "commit": commit,
        "env_id": "gym_pybullet_drones.envs.HoverAviary.HoverAviary + train_ppo.HoverItemEnv",
        "env_kwargs": ENV_KWARGS,
        "assets_files_sha256": "sha256:" + h.hexdigest(),   # sorted (relative path, file SHA-256) lines
        "env_class_sha256": env_class_sha256(),
        "pybullet": importlib.metadata.version("pybullet"),
    }


def write_env_snapshot(dest: Path) -> int:
    """Materialize the environment DatasetSnapshot (design rev 11 C3; 05 section 12.1): one shard per
    env_params_version, plus env_version.json. Refuses an existing non-empty directory."""
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} exists and is not empty")
    (dest / SHARD_DIR).mkdir(parents=True, exist_ok=True)
    shard = dest / SHARD_DIR / f"{ENV_PARAMS_VERSION}.jsonl"
    shard.write_text(json.dumps(ENV_PARAMS_V1, sort_keys=True) + "\n", encoding="utf-8")
    write_json(dest / ENV_VERSION_FILE, env_version_doc())
    print(json.dumps({"snapshot": str(dest), "shard": str(shard), "shard_sha256": sha256_file(shard)}, indent=2))
    return 0


def read_env_params(snapshot: Path) -> tuple[dict, Path]:
    shard = snapshot / SHARD_DIR / f"{ENV_PARAMS_VERSION}.jsonl"
    lines = [ln for ln in shard.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) != 1:
        raise SystemExit(f"{shard} must hold exactly one line (one env_params_version per shard)")
    params = json.loads(lines[0])
    if params != ENV_PARAMS_V1:
        raise SystemExit(f"{shard} is not the env_params_version {ENV_PARAMS_VERSION} this script implements")
    return params, shard


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_env_fn(env_params: dict, rank: int):
    def _init():
        import warnings

        warnings.filterwarnings("ignore", message=".*precision lowered by casting.*")
        cls = make_hover_item_env_class()
        return cls(env_params, ENV_KWARGS, rank)
    return _init


def build_vec_env(s: dict, env_params: dict):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [make_env_fn(env_params, i) for i in range(s["n_envs"])]
    if s["vec_env"] == "subproc":
        return SubprocVecEnv(fns, start_method="forkserver")
    return DummyVecEnv(fns)


def save_cut(model, out: Path, iteration: int, env_version: dict, extra: dict) -> Path:
    """One cut: model.zip (SB3 save, resume state), policy.pth (the zip's weights member, deployable),
    policy_meta.json, vecnormalize.pkl (VecNormalize.save) and its JSON mirror, env_version.json, then
    cut.json last (completeness marker, with every file's SHA-256)."""
    cut = out / f"checkpoint-{iteration}"
    if cut.exists():
        return cut
    tmp = out / f".checkpoint-{iteration}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    model.save(tmp / "model.zip")
    with zipfile.ZipFile(tmp / "model.zip") as z:
        (tmp / "policy.pth").write_bytes(z.read("policy.pth"))
    pol = model.policy
    write_json(tmp / "policy_meta.json", {
        "policy_class": f"{type(pol).__module__}.{type(pol).__name__}",
        "net_arch": model.policy_kwargs.get("net_arch"),
        "activation_fn": "torch.nn.Tanh",               # ActorCriticPolicy default
        "observation_space": {"shape": list(model.observation_space.shape)},
        "action_space": {"shape": list(model.action_space.shape), "low": -1.0, "high": 1.0},
        "stable_baselines3": importlib.metadata.version("stable-baselines3"),
        "deterministic_action": "mean of the Gaussian (predict(deterministic=True))",
    })
    vn = model.get_vec_normalize_env()
    vn.save(str(tmp / "vecnormalize.pkl"))
    write_json(tmp / "vecnormalize.json", {
        "obs_mean": vn.obs_rms.mean.tolist(), "obs_var": vn.obs_rms.var.tolist(), "obs_count": float(vn.obs_rms.count),
        "ret_mean": float(vn.ret_rms.mean), "ret_var": float(vn.ret_rms.var), "ret_count": float(vn.ret_rms.count),
        "clip_obs": vn.clip_obs, "clip_reward": vn.clip_reward, "epsilon": vn.epsilon, "gamma": vn.gamma,
        "norm_obs": vn.norm_obs, "norm_reward": vn.norm_reward,
    })
    write_json(tmp / ENV_VERSION_FILE, env_version)
    files = {name: sha256_file(tmp / name) for name in CUT_FILES}
    write_json(tmp / CUT_MARKER, {
        "schema": "gatedtrain_mm.sb3_ppo_cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "iteration": iteration, "num_timesteps": int(model.num_timesteps), "n_updates": int(model._n_updates),
        "files": files, "created_at": utc_now(), **extra,
    })
    os.replace(tmp, cut)
    return cut


def make_callbacks(out: Path, s: dict, first_iteration: int, env_version: dict, phase: str):
    from stable_baselines3.common.callbacks import BaseCallback

    rollout_size = s["n_steps"] * s["n_envs"]

    class GatedtrainCallback(BaseCallback):
        """Rollout log at each rollout end (before the update that uses it), cuts at rollout starts (after the
        previous update) and at training end, per-iteration timing."""

        def __init__(self):
            super().__init__(verbose=0)
            self.local = 0
            self.t_iter = None
            self.last_saved = None

        def _cut(self):
            it = first_iteration + self.local
            if self.last_saved != it:
                save_cut(self.model, out, it, env_version, {"phase": phase})
                self.last_saved = it

        def _on_rollout_start(self) -> None:
            now = time.time()
            if self.local > 0:
                append_jsonl(out / "memory.jsonl", {
                    "iteration": first_iteration + self.local, "utc": utc_now(),
                    "seconds": round(now - self.t_iter, 3),
                    "max_rss_mb_self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                    "max_rss_mb_children": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
                })
                if self.local % s["save_every"] == 0:
                    self._cut()
            self.t_iter = now
            self.t0 = self.model.num_timesteps

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            self.local += 1
            it = first_iteration + self.local
            per_env = self.training_env.env_method("pop_rollout_log")
            episodes = [e for env_log in per_env for e in env_log]
            steps = sum(e["steps"] for e in episodes)
            if steps != rollout_size or self.model.num_timesteps - self.t0 != rollout_size:
                raise RuntimeError(f"rollout {it}: logged {steps} steps, expected {rollout_size}")
            for e in episodes:
                e["payload"] = init_payload(ENV_PARAMS_VERSION, e["init"])
            append_jsonl(out / "rollouts.jsonl", {
                "iteration": it,
                "phase": phase,
                "num_timesteps_first": int(self.t0) + 1,     # cumulative env steps of this rollout, 1-based
                "num_timesteps_last": int(self.model.num_timesteps),
                "n_envs": s["n_envs"], "n_steps": s["n_steps"],
                "env_params_version": ENV_PARAMS_VERSION,
                "shard": f"{ENV_PARAMS_VERSION}.jsonl",
                "episodes": episodes,
                "utc": utc_now(),
            })

        def _on_training_end(self) -> None:
            if self.local > 0:
                append_jsonl(out / "memory.jsonl", {
                    "iteration": first_iteration + self.local, "utc": utc_now(),
                    "seconds": round(time.time() - self.t_iter, 3),
                    "max_rss_mb_self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                    "max_rss_mb_children": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
                })
                self._cut()

    return GatedtrainCallback()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--phase", choices=["base", "continue"])
    p.add_argument("--profile", choices=["smoke", "full"])
    p.add_argument("--out", type=Path)
    p.add_argument("--env-snapshot", type=Path, default=Path(DEFAULT_SNAPSHOT))
    p.add_argument("--base", type=Path, help="continue phase: the base cut directory (checkpoint-<I>)")
    p.add_argument("--iterations", type=int)
    p.add_argument("--save-every", type=int)
    p.add_argument("--seed", type=int)
    for key in ("n_envs", "n_steps", "batch_size", "n_epochs", "torch_threads"):
        p.add_argument("--" + key.replace("_", "-"), type=int)
    p.add_argument("--vec-env", choices=["subproc", "dummy"])
    p.add_argument("--write-env-snapshot", type=Path, metavar="DIR")
    args = p.parse_args(argv)
    if args.write_env_snapshot is None:
        if not (args.phase and args.profile and args.out):
            p.error("--phase, --profile and --out are required")
        if (args.phase == "continue") != (args.base is not None):
            p.error("--base is required with --phase continue and refused with --phase base")
    return args


def resolve_settings(args) -> dict:
    s = dict(COMMON)
    s.update(PROFILES[(args.phase, args.profile)])
    for key in ("iterations", "save_every", "n_envs", "n_steps", "batch_size", "n_epochs", "torch_threads",
                "vec_env"):
        if getattr(args, key) is not None:
            s[key] = getattr(args, key)
    s["seed"] = args.seed if args.seed is not None else PHASE_SEED[args.phase]
    if (s["n_steps"] * s["n_envs"]) % s["batch_size"]:
        raise SystemExit("n_steps * n_envs must be a multiple of batch_size")
    if s["iterations"] % s["save_every"]:
        raise SystemExit("iterations must be a multiple of save_every so the last iteration is a cut")
    s["rollout_size"] = s["n_steps"] * s["n_envs"]
    s["total_timesteps"] = s["iterations"] * s["rollout_size"]
    return s


def read_base(base: Path) -> dict:
    marker = base / CUT_MARKER
    if not marker.is_file():
        raise SystemExit(f"{base} has no {CUT_MARKER}: not a complete cut")
    doc = json.loads(marker.read_text(encoding="utf-8"))
    for name, want in doc["files"].items():
        if sha256_file(base / name) != want:
            raise SystemExit(f"{base / name} differs from its SHA-256 in {CUT_MARKER}")
    return doc


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.write_env_snapshot is not None:
        return write_env_snapshot(args.write_env_snapshot.resolve())

    out = args.out.resolve()
    attempt = prepare_out(out)
    write_env_freeze(out)                         # before torch is imported (MM-33)
    s = resolve_settings(args)
    env_params, shard = read_env_params(args.env_snapshot)
    base_doc = read_base(args.base) if args.base is not None else None
    first_iteration = int(base_doc["iteration"]) if base_doc else 0
    t_start = time.time()

    import torch

    torch.set_num_threads(s["torch_threads"])
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    env_version = env_version_doc()
    snap_version = json.loads((args.env_snapshot / ENV_VERSION_FILE).read_text(encoding="utf-8"))
    if snap_version != env_version:
        raise SystemExit(f"{args.env_snapshot / ENV_VERSION_FILE} does not describe this runtime's environment")

    run_args = {
        "schema": "gatedtrain_mm.sb3_ppo_run_args", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "phase": args.phase, "profile": args.profile, "attempt": attempt,
        "settings": s,
        "seed": s["seed"],
        "env_seeds": [s["seed"] + i for i in range(s["n_envs"])],
        "seed_derivation": ("PPO(seed) and PPO.load(..., seed=seed) call set_random_seed(seed) in _setup_model "
                            "(stable_baselines3/common/on_policy_algorithm.py:115-117): random, numpy and torch "
                            "seeded, and VecEnv.seed(seed) makes env i reset first with seed + i "
                            "(common/vec_env/base_vec_env.py:308); later automatic resets pass no seed, so each "
                            "env's initial states are one stream of its np_random (gymnasium/core.py:169-171, "
                            "utils/seeding.py:39 SeedSequence(seed) -> PCG64)"),
        "env_snapshot": str(args.env_snapshot),
        "env_params_file": str(shard),
        "env_params_sha256": sha256_file(shard),
        "env_params_version": ENV_PARAMS_VERSION,
        "env_version_sha256": sha256_file(args.env_snapshot / ENV_VERSION_FILE),
        "env_kwargs": ENV_KWARGS,
        "base_checkpoint": str(args.base) if args.base else None,
        "base_cut": base_doc,
        "first_iteration": first_iteration + 1,
        "last_iteration": first_iteration + s["iterations"],
        "consumption_log": "rollouts.jsonl",
        "consumption_semantics": ("line I lists every episode with at least one env step in rollout I; PPO update "
                                  "I (OnPolicyAlgorithm.train after collect_rollouts, on_policy_algorithm.py:323-337) "
                                  "uses exactly that rollout; cut checkpoint-I holds the weights after update I"),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "created_at": utc_now(),
    }
    write_json(out / "run_args.json", run_args)

    venv = build_vec_env(s, env_params)
    if base_doc is None:
        venv = VecNormalize(venv, norm_obs=s["norm_obs"], norm_reward=s["norm_reward"], clip_obs=s["clip_obs"],
                            gamma=s["gamma"])
        model = PPO("MlpPolicy", venv, learning_rate=s["learning_rate"], n_steps=s["n_steps"],
                    batch_size=s["batch_size"], n_epochs=s["n_epochs"], gamma=s["gamma"],
                    gae_lambda=s["gae_lambda"], clip_range=s["clip_range"], ent_coef=s["ent_coef"],
                    vf_coef=s["vf_coef"], max_grad_norm=s["max_grad_norm"],
                    policy_kwargs={"net_arch": s["net_arch"]}, seed=s["seed"], device="cpu", verbose=0)
    else:
        venv = VecNormalize.load(str(args.base / "vecnormalize.pkl"), venv)
        venv.training = True
        venv.norm_reward = s["norm_reward"]
        # `seed` in kwargs replaces the stored one before _setup_model seeds the env (base_class.py:738-740), so the
        # continue phase's envs draw other initial states than the base phase's.
        model = PPO.load(str(args.base / "model.zip"), env=venv, device="cpu", seed=s["seed"])
        for key in ("n_steps", "batch_size", "n_epochs"):
            if getattr(model, key) != s[key]:
                raise SystemExit(f"base model {key}={getattr(model, key)} differs from settings {s[key]}")
    cb = make_callbacks(out, s, first_iteration, env_version, args.phase)
    model.learn(total_timesteps=s["total_timesteps"], callback=cb, reset_num_timesteps=base_doc is None)
    venv.close()
    append_jsonl(out / "attempts.jsonl", {"attempt": attempt, "finished_at": utc_now(),
                                          "last_iteration": first_iteration + cb.local,
                                          "num_timesteps": int(model.num_timesteps),
                                          "wall_seconds": round(time.time() - t_start, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
