#!/usr/bin/env python
"""Per-item evaluation of an SB3 PPO hover cut for prototype area "control" (prototype, unpromotable).

Evaluators `drones.sb3_items@0.1` (deciding, `hover_success`) and `drones.sb3_return_items@0.1` (supporting,
`hover_score`, the scaled training reward: T0 under design 05 section 6.1, never deciding). An item is one registered initial state of the hover task (design 05 section 3.3:
"an episode seed with environment parameters"); its payload is the exact initial state (x, y, z, vz) the env
applies, so the result does not depend on any RNG. Per item: one episode with the deterministic policy
(`predict(deterministic=True)`, the Gaussian mean), observations normalized by the cut's frozen VecNormalize
statistics (`training=False`, `norm_reward=False`), the environment's raw reward.

Deciding metric `hover_success` (T3; higher is better, range [0, 1]; not a function of the reward): the fraction
of the second half of the episode's control steps (steps 121 .. 242 of at most 242) in which the drone is inside the
non-truncation bounds (|x|, |y| <= 1.5, z <= 2, |roll|, |pitch| <= 0.4; HoverAviary.py:110-113) and its altitude
error |z - 1| is at most TOL_Z = 0.06 m, the radius of the CF2X collision cylinder (assets/cf2x.urdf:34, read by
BaseAviary.py:1001 as COLLISION_R): the target height lies within the airframe's own extent. Only the altitude is
scored because the `one_d_rpm` action sets one collective thrust (BaseRLAviary.py:223-224): with zero initial
attitude no policy can move the drone horizontally, so the item's horizontal offset is fixed by the item and is not
a property of the policy. A crash (truncation before step 242) counts 0 for its remaining steps; a termination on
reaching the target exactly (HoverAviary.py:93-94) counts its remaining steps as successes.

Supporting metric `hover_score` (higher is better, range [0, 1]): the episode return divided by its maximum, 2 x 242
(HoverAviary reward max(0, 2 - d^4) per control step, gym_pybullet_drones/envs/HoverAviary.py:77-79; at most 242
control steps before the time-limit truncation, :114). An episode cut short by the out-of-bounds or tilt
truncation (:110-113) scores 0 for its remaining steps (a crash is a score, MM-4); an episode that terminates on
reaching the target exactly (:93-94) is credited 2 for its remaining steps. Sub-metrics (descriptive only):
`crashed`, `steps`, `hover_success`, `hover_score`, `success_3d_frac` (the same rule on the 3-D distance, for
diagnosis only), `within_10cm_frac`, `final_dist`, `final_z_err`, `mean_dist`, `episode_return`. `--metric` picks
which of the two metrics is the arm's `score`; both are always in `sub_metrics`.

Two modes:
  python eval_hover.py --make-items --n N --item-seed S --out ITEMS_DIR [--env-snapshot DIR]
      writes the registered item set ITEMS_DIR/items.jsonl (and item_set_meta.json)
  python eval_hover.py --checkpoint CUT --items ITEMS_DIR --arm-out ARM --eval-seed SEED
                       [--model-hash H] [--item-set-hash H] [--limit N] [--workers W]
      writes an arm directory ARM/items.jsonl (MULTIMODAL-SPEC MM-1) and ARM/eval_meta.json (MM-2)

The arm needs two gatedtrain hashes this runtime does not compute (the adapter's `model_hash` of the cut, MM-17, and
the item set's manifest hash, MM-6); pass them with --model-hash / --item-set-hash, or let gatedtrain fill
`eval_meta.json` (README.md "What gatedtrain needs"). Without them both fields are written as null.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

ADAPTER = "sb3_ppo"
# metric -> (evaluator id, scorer version); `hover_success` decides, `hover_score` supports (T0).
METRICS = {"hover_success": ("drones.sb3_items@0.1", "eval_hover@0.2"),
           "hover_score": ("drones.sb3_return_items@0.1", "eval_hover@0.2")}
DIRECTION = "higher_is_better"
SCORE_RANGE = [0, 1]
ERROR_CLASSES = ("simulator_error",)
MAX_STEPS = 242               # (EPISODE_LEN_SEC * PYB_FREQ) / PYB_STEPS_PER_CTRL + 2: truncation at step 242
MAX_REWARD = 2.0
TOL = 0.10                    # within_10cm_frac (descriptive)
TOL_Z = 0.06                  # hover_success: CF2X collision cylinder radius (assets/cf2x.urdf:34)
SUCCESS_FIRST_STEP = 121      # second half of the 242 control steps
SUCCESS_STEPS = MAX_STEPS - SUCCESS_FIRST_STEP + 1   # 122
BOUNDS = {"xy": 1.5, "z_max": 2.0, "tilt": 0.4}      # HoverAviary.py:110-113
ITEM_NAMESPACE = 0x5EED17E5   # item initial states: SeedSequence([ITEM_NAMESPACE, item_seed]); training envs use
                              # SeedSequence(int seed), a different entropy, and the check is exact anyway
ENV_PARAMS_VERSION = "hover-init-v1"
SHARD_DIR = "shards"
ENV_VERSION_FILE = "env_version.json"
SAMPLER = {"policy": "deterministic (Gaussian mean)", "vecnormalize": {"training": False, "norm_reward": False},
           "episodes_per_item": 1, "max_steps": MAX_STEPS}


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


def init_payload(version: str, init: dict) -> str:
    """Same text as train_ppo.py `init_payload` (the exact-overlap check compares these strings)."""
    return f"{version} x={init['x']:.4f} y={init['y']:.4f} z={init['z']:.4f} vz={init['vz']:.4f}"


# The block below is a byte-identical copy of train_ppo.py's; `env_class_sha256` hashes it and the evaluation
# refuses a cut whose env_version.json names another hash.
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
    text = Path(__file__).resolve().read_text(encoding="utf-8")
    start, end = text.index("# BEGIN HoverItemEnv\n"), text.index("# END HoverItemEnv\n")
    return sha256_bytes(text[start:end].encode("utf-8"))


def read_env_params(snapshot: Path) -> dict:
    shard = snapshot / SHARD_DIR / f"{ENV_PARAMS_VERSION}.jsonl"
    lines = [ln for ln in shard.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) != 1:
        raise SystemExit(f"{shard} must hold exactly one line")
    return json.loads(lines[0])


# ---------------------------------------------------------------------------
# Item set
# ---------------------------------------------------------------------------

def make_items(n: int, item_seed: int, snapshot: Path, out: Path) -> int:
    import numpy as np

    params = read_env_params(snapshot)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(np.random.SeedSequence([ITEM_NAMESPACE, item_seed]))
    seen, rows = set(), []
    while len(rows) < n:
        init = {k: round(float(rng.uniform(*params["init_ranges"][k])), params["decimals"])
                for k in params["init_order"]}
        payload = init_payload(params["env_params_version"], init)
        if payload in seen:
            continue
        seen.add(payload)
        rows.append({"item": f"{params['env_params_version']}-{len(rows):04d}", "env_params_version":
                     params["env_params_version"], "init": init, "payload": payload})
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (out / "items.jsonl").write_text(text, encoding="utf-8")
    meta = {"schema": "ptc_mm.drones_item_set", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
            "n": n, "item_seed": item_seed, "generator": f"numpy SeedSequence([{ITEM_NAMESPACE:#x}, item_seed]) "
            "-> default_rng; per item uniform draws in init_order, rounded to decimals; duplicates skipped",
            "numpy": np.__version__, "env_params_sha256": sha256_file(snapshot / SHARD_DIR /
                                                                      f"{ENV_PARAMS_VERSION}.jsonl"),
            "items_sha256": sha256_bytes(text.encode("utf-8"))}
    (out / "item_set_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


def read_items(items_dir: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in (items_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    ids = [r["item"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("item ids are not unique")
    for r in rows:
        if init_payload(r["env_params_version"], r["init"]) != r["payload"]:
            raise SystemExit(f"item {r['item']}: payload does not match its init")
    return rows


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _worker_init(cut: str, params: dict, env_kwargs: dict, threads: int, eval_seed: int, metric: str) -> None:
    import warnings

    warnings.filterwarnings("ignore", message=".*precision lowered by casting.*")
    import torch

    torch.set_num_threads(threads)
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    set_random_seed(eval_seed)
    cls = make_hover_item_env_class()
    env = cls(params, env_kwargs, rank=0)
    vn = VecNormalize.load(str(Path(cut) / "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
    vn.training = False
    vn.norm_reward = False
    model = PPO.load(str(Path(cut) / "model.zip"), device="cpu")
    _WORKER.update(env=env, vn=vn, model=model, metric=metric)


def run_item(item: dict) -> dict:
    import numpy as np

    env, vn, model = _WORKER["env"], _WORKER["vn"], _WORKER["model"]
    target = np.array([0.0, 0.0, 1.0])
    try:
        obs, _ = env.reset(seed=None, options={"init": item["init"]})
        ret, steps, within, dists = 0.0, 0, 0, []
        succ = succ3d = 0
        ok = ok3d = False
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            ret += float(reward)
            pos, rpy = env.pos[0], env.rpy[0]
            d = float(np.linalg.norm(target - pos))
            dists.append(d)
            within += d < TOL
            inside = (abs(pos[0]) <= BOUNDS["xy"] and abs(pos[1]) <= BOUNDS["xy"] and pos[2] <= BOUNDS["z_max"]
                      and abs(rpy[0]) <= BOUNDS["tilt"] and abs(rpy[1]) <= BOUNDS["tilt"])
            ok = bool(inside and abs(pos[2] - 1.0) <= TOL_Z)
            ok3d = bool(inside and d <= TOL_Z)
            if steps >= SUCCESS_FIRST_STEP:
                succ += ok
                succ3d += ok3d
            if steps > MAX_STEPS:
                raise RuntimeError(f"episode exceeded {MAX_STEPS} steps")
        crashed = bool(truncated and steps < MAX_STEPS)
        early_done = bool(terminated and not truncated and steps < MAX_STEPS)
        credit = MAX_REWARD * (MAX_STEPS - steps) if early_done else 0.0
        if early_done:                            # target reached exactly: the remaining steps count as hovering
            rest = MAX_STEPS - max(steps, SUCCESS_FIRST_STEP - 1)
            succ += rest * ok
            succ3d += rest * ok3d
        hover_score = min(max((ret + credit) / (MAX_REWARD * MAX_STEPS), 0.0), 1.0)
        hover_success = succ / SUCCESS_STEPS
        if not (math.isfinite(hover_score) and 0.0 <= hover_success <= 1.0):
            raise FloatingPointError(f"scores {hover_score}, {hover_success}")
        metrics = {"hover_success": hover_success, "hover_score": hover_score}
        return {"item": item["item"], "score": metrics[_WORKER["metric"]], "error": None, "sub_metrics": {
            "hover_success": hover_success, "hover_score": hover_score,
            "success_3d_frac": succ3d / SUCCESS_STEPS,
            "episode_return": round(ret, 6), "steps": steps, "crashed": int(crashed),
            "within_10cm_frac": within / MAX_STEPS, "final_dist": round(dists[-1], 6),
            "final_z_err": round(abs(float(env.pos[0][2]) - 1.0), 6),
            "mean_dist": round(sum(dists) / len(dists), 6)}}
    except Exception as exc:                      # the simulator or the policy raised: evaluator error (MM-4)
        return {"item": item["item"], "score": None, "error": {"class": "simulator_error",
                                                                 "message": f"{type(exc).__name__}: {exc}"[:500]},
                "sub_metrics": None}


def check_cut(cut: Path) -> dict:
    marker = json.loads((cut / "cut.json").read_text(encoding="utf-8"))
    for name, want in marker["files"].items():
        if sha256_file(cut / name) != want:
            raise SystemExit(f"{cut / name} differs from its SHA-256 in cut.json")
    with zipfile.ZipFile(cut / "model.zip") as z:
        if sha256_bytes(z.read("policy.pth")) != marker["files"]["policy.pth"]:
            raise SystemExit("model.zip policy.pth is not the cut's policy.pth: the evaluated weights would differ")
    env_version = json.loads((cut / ENV_VERSION_FILE).read_text(encoding="utf-8"))
    if env_version.get("env_class_sha256") != env_class_sha256():
        raise SystemExit("the cut was trained with another HoverItemEnv than this evaluator's")
    return marker


def evaluate(args) -> int:
    cut = args.checkpoint.resolve()
    marker = check_cut(cut)
    env_version = json.loads((cut / ENV_VERSION_FILE).read_text(encoding="utf-8"))
    snap_version_sha = sha256_file(args.env_snapshot / ENV_VERSION_FILE)
    if sha256_file(cut / ENV_VERSION_FILE) != snap_version_sha:
        raise SystemExit("the cut's env_version.json is not the registered evaluation environment's")
    params = read_env_params(args.env_snapshot)
    items = read_items(args.items)
    if args.limit:
        items = items[: args.limit]
    arm = args.arm_out.resolve()
    if arm.exists() and any(arm.iterdir()):
        raise SystemExit(f"{arm} exists and is not empty")
    arm.mkdir(parents=True, exist_ok=True)
    lock = HERE / "uv.lock"
    runtime = {"python": sys.version.split()[0], "executable": sys.executable,
               "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
               "script_sha256": sha256_file(Path(__file__).resolve()), "env_class_sha256": env_class_sha256(),
               "checkpoint": str(cut), "items": str(args.items.resolve()),
               "items_sha256": sha256_file(args.items / "items.jsonl"), "workers": args.workers,
               "threads": args.threads, "uid": os.getuid(), "gid": os.getgid()}
    (arm / "env-runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    t0 = time.time()
    init_args = (str(cut), params, env_version["env_kwargs"], args.threads, args.eval_seed, args.metric)
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("forkserver").Pool(args.workers, initializer=_worker_init, initargs=init_args) as pool:
            results = pool.map(run_item, items, chunksize=max(1, len(items) // (args.workers * 4)))
    else:
        _worker_init(*init_args)
        results = [run_item(it) for it in items]
    wall = time.time() - t0

    by_item = {it["item"]: it for it in items}
    lines = []
    for r in results:
        it = by_item[r["item"]]
        lines.append({"item": r["item"], "repeat": 0, "seed": args.eval_seed, "score": r["score"],
                      "error": r["error"], "sub_metrics": r["sub_metrics"], "output_ref": None,
                      "item_meta": {"init": it["init"], "env_params_version": it["env_params_version"]}})
    (arm / "items.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines), encoding="utf-8")
    evaluator_id, scorer_version = METRICS[args.metric]
    sampler_hash = sha256_bytes(json.dumps(SAMPLER, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    meta = {
        "schema": "gatedtrain.item_eval", "schema_version": "1.0",
        "evaluator_id": evaluator_id, "adapter": ADAPTER, "metric": args.metric, "direction": DIRECTION,
        "score_range": SCORE_RANGE,
        "checkpoint": {"model_hash": args.model_hash, "base_model_hash": None},
        "normalizer_sha256": sha256_file(cut / "vecnormalize.pkl"),
        "env_version_sha256": snap_version_sha,
        "item_set_hash": args.item_set_hash,
        "eval_seeds": [args.eval_seed],
        "sampler": SAMPLER, "sampler_hash": sampler_hash,
        "scorer": {"name": args.metric, "version": scorer_version, "weights_sha256": None,
                   "code_sha256": sha256_file(Path(__file__).resolve())},
        "framework": {"name": "stable-baselines3", "version": importlib.metadata.version("stable-baselines3")},
        "created_at": utc_now(),
    }
    (arm / "eval_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scores = [x["score"] for x in lines if x["score"] is not None]
    n = len(scores)
    mean = sum(scores) / n if n else float("nan")
    sd = math.sqrt(sum((x - mean) ** 2 for x in scores) / (n - 1)) if n > 1 else float("nan")
    summary = {"cut": str(cut), "iteration": marker["iteration"], "num_timesteps": marker["num_timesteps"],
               "items": len(lines), "errors": len(lines) - n, "mean": mean, "sd": sd,
               "min": min(scores) if scores else None, "max": max(scores) if scores else None,
               "crashed": sum(x["sub_metrics"]["crashed"] for x in lines if x["sub_metrics"]),
               "distinct_scores": len({round(x, 9) for x in scores}), "wall_seconds": round(wall, 2),
               "metric": args.metric, "frac_at_0": sum(x == 0 for x in scores) / n if n else None,
               "frac_at_1": sum(x == 1 for x in scores) / n if n else None,
               "missing_hashes": [k for k, v in (("model_hash", args.model_hash),
                                                  ("item_set_hash", args.item_set_hash)) if v is None]}
    (arm / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--make-items", action="store_true")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--item-seed", type=int, default=20261112)
    p.add_argument("--out", type=Path)
    p.add_argument("--env-snapshot", type=Path, default=HERE / "env-snapshot")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--items", type=Path, default=HERE / "items")
    p.add_argument("--arm-out", type=Path)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--model-hash")
    p.add_argument("--item-set-hash")
    p.add_argument("--metric", choices=sorted(METRICS), default="hover_success",
                   help="the arm's score: hover_success (deciding) or hover_score (supporting, T0)")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)
    if args.make_items:
        if args.out is None:
            p.error("--make-items needs --out")
    elif args.checkpoint is None or args.arm_out is None:
        p.error("--checkpoint and --arm-out are required")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.make_items:
        return make_items(args.n, args.item_seed, args.env_snapshot, args.out.resolve())
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
