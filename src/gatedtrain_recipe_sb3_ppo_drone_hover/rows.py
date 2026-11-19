"""Registry rows of adapter `sb3_ppo`: SB3 PPO drone hover.

The training row, the evaluator rows and the adapter class, loaded by `gatedtrain.ext.registry` through the
`gatedtrain.adapters` entry point (MULTIMODAL-SPEC section 6).

Paths: the runtime directory is this repository's `runtime/` (scripts, its `.venv`, the env snapshot and the items;
mounted read-only at its own host path), or `GATEDTRAIN_RECIPE_SB3_PPO_DRONE_HOVER_RUNTIME`. The interpreter tree is the
one the runtime venv was made from (its `pyvenv.cfg` `home`); before `uv sync` there is none and ext-launch refuses. The
base runs' outputs lie under the host's output root (`{output_root}`, filled in by gatedtrain from the host layout).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from gatedtrain.ext.rowkit import CONTAINER_PATHS

NAME = "sb3_ppo"
RUNTIME_ENV = "GATEDTRAIN_RECIPE_SB3_PPO_DRONE_HOVER_RUNTIME"
OUT_NAME = "sb3-ppo-drone-hover"                     # the recipe's directory under the host's output root


def runtime_dir() -> str:
    """The runtime directory: the override, else `<repository>/runtime` (an editable or source checkout)."""
    return os.environ.get(RUNTIME_ENV) or str(Path(__file__).resolve().parents[2] / "runtime")


def python_tree(runtime: str) -> str | None:
    """The interpreter tree the runtime venv links to (`pyvenv.cfg` `home` = `<tree>/bin`), or None without a venv."""
    try:
        cfg = (Path(runtime) / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in cfg.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "home" and value.strip():
            return str(Path(value.strip()).parent)
    return None


RUNTIME = runtime_dir()
TREE = python_tree(RUNTIME)
PY_VERSION = Path(TREE).name.split("-")[1].rsplit(".", 1)[0] if TREE else None    # cpython-3.12.16-... -> 3.12

TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}

# Area "control" (MULTIMODAL-SPEC section 6 row 4; runtime/README.md "What gatedtrain needs" item 8). Two phases, two launches: the base run trains from scratch, the continuation loads a base
# cut; ext-collect gates the continuation against that cut (EXT-101).
TRAINER_ROWS["sb3_ppo"] = {
    "uses": ("rl.sb3_ppo@0.1",),
    "pins": ("stable-baselines3", "gymnasium", "gym-pybullet-drones", "pybullet"),   # from env-freeze.txt
    "outputs": ("checkpoint-<iteration>/ cuts with cut.json written last", "run_args.json", "rollouts.jsonl"),
    # EXT-99: model_hash components (MM-17, MM-19): weights, policy description, normalizer and env version.
    "components": ("policy.pth", "policy_meta.json", "vecnormalize.pkl", "vecnormalize.json", "env_version.json"),
    # EXT-102 (chosen 2026-11-12): observed rollout log, one row per (iteration, shard); a log that
    # does not match the run refuses, there is no fallback.
    "consumption": {"target": "iteration", "fallback": None},
    "gpu_h_estimate": 0.0,                           # CPU only (README "CPU, not GPU"; EXT-105)
    "evaluators": ("drones.sb3_items@0.1", "drones.sb3_return_items@0.1"),   # deciding first (EXT-105)
    "framework": "stable-baselines3",
    # EXT-100: the interpreter, the runtime directory (scripts, venv, env snapshot, items) and the base runs'
    # outputs, which lie under the output root (EXT-104).
    "input_roots": tuple(([str(Path(TREE).parent)] if TREE else []) + [RUNTIME, "{output_root}/" + OUT_NAME]),
    # The base phase has no spec yet (its cut is the spec's base), so --spec is optional at ext-launch; when it is
    # given, the base cut and the env-params shard directory are checked against the mounts (EXT-101, EXT-103).
    "spec_binding": False,
    # README "Mounts and command shape": all of them, none a credential.
    "env_names": ("HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "XDG_CACHE_HOME"),
    "scan_skip": tuple([f"{RUNTIME}/.venv"] + ([f"{TREE}/lib/python{PY_VERSION}/site-packages"] if TREE else [])),
    "interpreter_trees": (TREE,) if TREE else (),
    "interpreter": f"{RUNTIME}/.venv/bin/python",                                     # README "Request shapes"
    "container_paths": CONTAINER_PATHS,
    "env_paths": ("HOME", "XDG_CACHE_HOME"),
}

# Area "control" (MULTIMODAL-SPEC section 6 row 4, with the deviations accepted 2026-11-12: a bounded deciding metric
# and N = 300). Scorer: eval_hover.py, code_sha256 its SHA-256 (EXT-98). An item is one registered initial state
# (05 section 3.3); its payload text is the exact-overlap key against the training rollouts (EXT-107).
_DRONES = {
    "direction": "higher_is_better",
    "score_range": (0, 1),
    "binary": False,
    "error_classes": ("simulator_error",),               # MM-4: the simulator or the policy raised
    "score_outcomes": ("crash", "below_tolerance", "timeout"),   # MM-4: the policy's faults are scores
    "payload_text_field": "payload",
    "item_payload": ("init", "payload"),
    "item_payload_objects": ("init",),                   # EXT-98: `init` is a JSON object, the others strings
    "normalizer": True,                                  # MM-19: the arm names the cut's vecnormalize.pkl
    "env_version": True,                                 # MM-19: the arm names the registered env_version.json
    # EXT-106: the evaluation sampler of a deterministic policy, exactly these keys (eval_hover.py SAMPLER).
    "sampler_keys": ("episodes_per_item", "max_steps", "policy", "vecnormalize"),
    "framework": "stable-baselines3",
    "samples": "300, 1",
}
EVAL_HOVER_SHA256 = "sha256:e067719e6aa6418dec61afa86f82502d5d89ebdb44e48de0d748fae6225960a7"
EVALUATOR_ROWS["drones.sb3_items@0.1"] = {
    **_DRONES, "metric": "hover_success", "tier": "T3",  # a simulator measurement, not the reward (05 section 6.1)
    "scorer": {"name": "hover_success", "version": "eval_hover@0.2", "weights_sha256": None,
               "code_sha256": EVAL_HOVER_SHA256}}
EVALUATOR_ROWS["drones.sb3_return_items@0.1"] = {
    **_DRONES, "metric": "hover_score", "tier": "T0",    # the training reward scaled: never deciding (05 section 6.1)
    "scorer": {"name": "hover_score", "version": "eval_hover@0.2", "weights_sha256": None,
               "code_sha256": EVAL_HOVER_SHA256}}

PLUGIN = {"adapters": {NAME: "gatedtrain_recipe_sb3_ppo_drone_hover.adapter:Sb3PpoAdapter"},
          "trainer_rows": TRAINER_ROWS, "evaluator_rows": EVALUATOR_ROWS}
