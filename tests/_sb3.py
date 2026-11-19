"""Area "control" test helpers (verifier-owned; tests/README.md section 11.19, EXT-98 to EXT-111).

Written from the runtime's README (`runtime/README.md`, "Outputs of
`train_ppo.py`", "Cut layout", "Consumption log", "What gatedtrain needs", read 2026-11-12) and the sanity outputs of
2026-11-12, not from gatedtrain. Small realistic pieces are copied into `fixtures/ext/sb3/`: the env snapshot's
`env_version.json` and shard `hover-init-v1.jsonl` (byte copies of `area2-drones/env-snapshot/`), a cut's
`policy_meta.json` and a run's `env-freeze.txt` (from the continue sanity run from base cut 8). Weights and the
pickled normalizer are byte stand-ins: nothing in gatedtrain opens them (MM-12), it hashes them.

Layout of `Sb3Scenario(root)`:
- `<root>/area2-drones/` the runtime directory stand-in (mounted read-only at `/inputs/a2`, EXT-112): `train_ppo.py`, `.venv/bin/python`
  (stand-ins, never executed), `env-snapshot/{env_version.json, shards/hover-init-v1.jsonl}`;
- `<root>/items/items.jsonl`: N items `hover-init-v1-0000` ..., each `{item, env_params_version, init, payload}`;
- `<root>/out/area2-drones/base-1/`: the base-phase run (seed 1000, iterations 1 .. 2, cuts 1 and 2; cut 2 is the
  spec's base), `<root>/out/area2-drones/cont-1/`: the continuation from cut 2 (seed 2000, iterations 3 .. 6, cuts 4
  and 6), each with run_args.json, rollouts.jsonl, env-freeze.txt, env-runtime.json, memory.jsonl, attempts.jsonl;
- spec, registration (gate on `hover_success`: superiority, margin 0, sigma 0.39 pilot, alpha 0.05, N 300, K 1),
  trainer.yaml, two arms, launch record (gpus false, gpu_est_h 0), smoke record, and the host work's GPU log.
Rollouts are small (2 envs x 8 steps = 16 env steps per iteration); training payloads are drawn with another seed
than the items and checked disjoint.
"""

from __future__ import annotations

import hashlib
import os
import importlib.metadata
import json
import pwd
import random
import shutil
import unittest
import zipfile
from pathlib import Path
from typing import Any, Callable

import yaml

from gatedtrain.testing.ext import (ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_INTERPRETER, ENV_INTERPRETER_TREES,
                                    ENV_OUTPUT_ROOT, IMAGE_ID, gpu_step_script, launch_label, host_config_sha256, sha_of,
                                    write_canonical)
from gatedtrain.testing.support import canonical_bytes, file_sha256, manifest_entries, manifest_hash, run_cli
from gatedtrain.ext.arm import META_SCHEMA
from gatedtrain.ext.launch import RECORD_SCHEMA
from gatedtrain.ext.smoke import SMOKE_SCHEMA

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]

SB3 = FIXTURES / "sb3"
ADAPTER = "sb3_ppo"
TRAINER_ID = "rl.sb3_ppo@0.1"
EVALUATOR_ID = "drones.sb3_items@0.1"
SUPPORT_ID = "drones.sb3_return_items@0.1"
# The TRL GRPO recipe's adapter and evaluator row (another distribution, gatedtrain-recipe-trl-grpo-gsm8k): tests that
# compare against them skip when it is not installed.
TRL_ADAPTER = "trl_grpo"
TRL_EVALUATOR_ID = "gsm8k.nemo_evaluator_items@0.1"
HAS_TRL = any(ep.name == TRL_ADAPTER for ep in importlib.metadata.entry_points(group="gatedtrain.adapters"))
NO_TRL = "needs the TRL GRPO recipe (gatedtrain-recipe-trl-grpo-gsm8k, adapter trl_grpo), not installed"
METRIC = "hover_success"
EVAL_HOVER_SHA256 = "sha256:e067719e6aa6418dec61afa86f82502d5d89ebdb44e48de0d748fae6225960a7"
EVAL_HOVER = REPO / "runtime" / "eval_hover.py"
SCORER = {"name": "hover_success", "version": "eval_hover@0.2", "weights_sha256": None, "code_sha256": EVAL_HOVER_SHA256}
SAMPLER = {"policy": "deterministic (Gaussian mean)", "vecnormalize": {"training": False, "norm_reward": False},
           "episodes_per_item": 1, "max_steps": 242}                  # eval_hover.py SAMPLER
SEEDS = [0]
CUT_FILES = ("model.zip", "policy.pth", "policy_meta.json", "vecnormalize.pkl", "vecnormalize.json",
             "env_version.json")
COMPONENTS = ("policy.pth", "policy_meta.json", "vecnormalize.pkl", "vecnormalize.json", "env_version.json")
SHARD = "hover-init-v1.jsonl"
N_ENVS, N_STEPS = 2, 8
ROLLOUT = N_ENVS * N_STEPS
LABEL = "mm-a2-drones-full"
RUNTIME_CONTAINER = "/inputs/a2"        # EXT-112: read-only inputs lie under /inputs/ (or at their host path)
BASE_CONTAINER = "/inputs/base"
SCRIPT_TEXT = "# fixture stand-in for runtime/train_ppo.py; never executed\n"
INTERPRETER_TEXT = "# fixture stand-in for the area-2 venv interpreter; never executed\n"
ENV_VERSION_SHA = file_sha256(SB3 / "env_version.json")
SHARD_SHA = file_sha256(SB3 / SHARD)
ENV_NAMES = ["HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "XDG_CACHE_HOME"]


def run_user() -> str:
    """The container's `<uid>:<gid>` in fixture requests and records: the uid running the tests and its primary group,
    so the output directories the tests create are owned by the container's uid (EXT-63) whoever runs them. The launch
    rules refuse root, so a test that needs it skips under uid 0."""
    uid = os.getuid()
    if uid == 0:
        raise unittest.SkipTest("the launch rules refuse a container running as root (uid 0); run the tests as a "
                                "non-root user")
    return f"{uid}:{pwd.getpwuid(uid).pw_gid}"


def payload(init: dict) -> str:
    """train_ppo.py / eval_hover.py `init_payload`."""
    return f"hover-init-v1 x={init['x']:.4f} y={init['y']:.4f} z={init['z']:.4f} vz={init['vz']:.4f}"


def draw_init(rng: random.Random) -> dict:
    return {"x": round(rng.uniform(-0.25, 0.25), 4), "y": round(rng.uniform(-0.25, 0.25), 4),
            "z": round(rng.uniform(0.15, 1.85), 4), "vz": round(rng.uniform(-0.5, 0.5), 4)}


def build_items(d: Path, n: int = 300, seed: int = 20261112) -> list[dict]:
    rng, rows, seen = random.Random(seed), [], set()
    while len(rows) < n:
        init = draw_init(rng)
        if payload(init) in seen:
            continue
        seen.add(payload(init))
        rows.append({"env_params_version": "hover-init-v1", "init": init, "item": f"hover-init-v1-{len(rows):04d}",
                     "payload": payload(init)})
    d.mkdir(parents=True, exist_ok=True)
    (d / "items.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return rows


def sb3_model_hash(cut: Path) -> str:
    """EXT-99: manifest hash of the deployable components only (not model.zip, not cut.json)."""
    entries = [e for e in manifest_entries(cut) if e["path"] in COMPONENTS]
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


def build_cut(out: Path, iteration: int, *, phase: str, tag: str = "", timesteps: int | None = None) -> Path:
    """`checkpoint-<iteration>/` as train_ppo.py save_cut writes it, `cut.json` last."""
    d = out / f"checkpoint-{iteration}"
    d.mkdir(parents=True)
    weights = f"policy weights stand-in {phase} {iteration} {tag}\n".encode() * 8
    (d / "policy.pth").write_bytes(weights)
    with zipfile.ZipFile(d / "model.zip", "w", zipfile.ZIP_STORED) as z:
        z.writestr("data", json.dumps({"n_steps": N_STEPS, "seed": 1000 if phase == "base" else 2000}))
        z.writestr("policy.pth", weights)
        z.writestr("policy.optimizer.pth", b"optimizer stand-in\n")
        z.writestr("_stable_baselines3_version", "2.9.2")
    shutil.copyfile(SB3 / "policy_meta.json", d / "policy_meta.json")
    (d / "vecnormalize.pkl").write_bytes(f"pickle stand-in, never unpickled {phase} {iteration} {tag}\n".encode())
    (d / "vecnormalize.json").write_text(json.dumps({"obs_count": float(iteration * ROLLOUT), "clip_obs": 10.0,
                                                     "norm_obs": True, "norm_reward": True}, indent=2) + "\n",
                                         encoding="utf-8")
    shutil.copyfile(SB3 / "env_version.json", d / "env_version.json")
    files = {n: file_sha256(d / n) for n in CUT_FILES}
    (d / "cut.json").write_text(json.dumps({
        "created_at": "2026-11-12T18:16:41Z", "files": files, "iteration": iteration, "n_updates": iteration * 10,
        "num_timesteps": timesteps if timesteps is not None else iteration * ROLLOUT, "phase": phase,
        "schema": "gatedtrain_mm.sb3_ppo_cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"]},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return d


def rollout_lines(first: int, last: int, *, phase: str, seed: int, rng: random.Random,
                  forbidden: set[str]) -> list[dict]:
    """One line per iteration as train_ppo.py GatedtrainCallback._on_rollout_end writes it; episodes of 8 steps each."""
    lines, n = [], 0
    for it in range(first, last + 1):
        eps = []
        for env in range(N_ENVS):
            init = draw_init(rng)
            while payload(init) in forbidden:
                init = draw_init(rng)
            eps.append({"env": env, "episode": n, "init": init, "payload": payload(init), "seed": seed + env,
                        "steps": N_STEPS})
        n += 1
        lines.append({"env_params_version": "hover-init-v1", "episodes": eps, "iteration": it, "n_envs": N_ENVS,
                      "n_steps": N_STEPS, "num_timesteps_first": (it - 1) * ROLLOUT + 1,
                      "num_timesteps_last": it * ROLLOUT, "phase": phase, "shard": SHARD,
                      "utc": "2026-11-12T18:15:09Z"})
    return lines


def write_jsonl(path: Path, docs: list[dict]) -> Path:
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")
    return path


def run_args(*, phase: str, first: int, last: int, seed: int, base_cut: dict | None,
             base_checkpoint: str | None, profile: str = "full", save_every: int = 2) -> dict:
    """train_ppo.py run_args.json (the keys ext-collect reads, as the sanity runs wrote them)."""
    return {
        "schema": "gatedtrain_mm.sb3_ppo_run_args", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "phase": phase, "profile": profile, "attempt": 1,
        "settings": {"n_envs": N_ENVS, "n_steps": N_STEPS, "batch_size": 8, "n_epochs": 10, "iterations": last - first + 1,
                     "save_every": save_every, "seed": seed, "rollout_size": ROLLOUT, "vec_env": "subproc"},
        "seed": seed, "env_seeds": [seed + i for i in range(N_ENVS)],
        "env_snapshot": f"{RUNTIME_CONTAINER}/env-snapshot",
        "env_params_file": f"{RUNTIME_CONTAINER}/env-snapshot/shards/{SHARD}",
        "env_params_sha256": SHARD_SHA, "env_params_version": "hover-init-v1",
        "env_version_sha256": ENV_VERSION_SHA,
        "base_checkpoint": base_checkpoint, "base_cut": base_cut,
        "first_iteration": first, "last_iteration": last, "consumption_log": "rollouts.jsonl",
        "device": "cpu", "torch_threads": 4, "created_at": "2026-11-12T18:15:06Z",
    }


def build_run(out: Path, *, phase: str, first: int, last: int, cuts: tuple[int, ...], seed: int,
              base: Path | None, rng: random.Random, forbidden: set[str], script_sha256: str,
              profile: str = "full") -> Path:
    out.mkdir(parents=True)
    base_cut = json.loads((base / "cut.json").read_text(encoding="utf-8")) if base is not None else None
    write_canonical(out / "run_args.json", run_args(phase=phase, first=first, last=last, seed=seed, base_cut=base_cut,
                                                     base_checkpoint=BASE_CONTAINER if base is not None else None,
                                                     profile=profile))
    shutil.copyfile(SB3 / "env-freeze.txt", out / "env-freeze.txt")
    (out / "env-runtime.json").write_text(json.dumps({"python": "3.12.16", "script_sha256": script_sha256,
                                                      "env_class_sha256": "sha256:" + "fa" * 32}, indent=2) + "\n",
                                          encoding="utf-8")
    write_jsonl(out / "rollouts.jsonl", rollout_lines(first, last, phase=phase, seed=seed, rng=rng, forbidden=forbidden))
    write_jsonl(out / "memory.jsonl", [{"iteration": i, "seconds": 2.2} for i in range(first, last + 1)])
    write_jsonl(out / "attempts.jsonl", [{"attempt": 1, "started_at": "2026-11-12T18:15:06Z"}])
    for it in cuts:
        build_cut(out, it, phase=phase)
    return out


def eval_meta(*, model: str | None, item_set: str | None, normalizer: str, **overrides) -> dict[str, Any]:
    """eval_hover.py's eval_meta.json (MM-2) for `--metric hover_success`."""
    meta = {"schema": META_SCHEMA, "schema_version": "1.0", "evaluator_id": EVALUATOR_ID, "adapter": ADAPTER,
            "metric": METRIC, "direction": "higher_is_better", "score_range": [0, 1],
            "checkpoint": {"model_hash": model, "base_model_hash": None}, "normalizer_sha256": normalizer,
            "env_version_sha256": ENV_VERSION_SHA, "item_set_hash": item_set, "eval_seeds": list(SEEDS),
            "sampler": dict(SAMPLER), "sampler_hash": sha_of(SAMPLER), "scorer": dict(SCORER),
            "framework": {"name": "stable-baselines3", "version": "2.9.2"}, "created_at": "2026-11-12T18:20:00Z"}
    meta.update(overrides)
    return meta


def build_arm(d: Path, items: list[dict], score_fn: Callable[[int], float], *, cut: Path, item_set: str | None,
              model: str | None = ..., meta: dict | None = None) -> Path:
    """eval_hover.py's arm: items.jsonl (sorted keys, one line per item, K = 1) and eval_meta.json."""
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, it in enumerate(items):
        s = float(score_fn(i))
        lines.append({"error": None, "item": it["item"], "item_meta": {"env_params_version": "hover-init-v1",
                                                                         "init": it["init"]},
                      "output_ref": None, "repeat": 0, "score": s, "seed": 0,
                      "sub_metrics": {"hover_success": s, "hover_score": 0.99, "crashed": 0, "steps": 242}})
    write_jsonl(d / "items.jsonl", lines)
    m = eval_meta(model=sb3_model_hash(cut) if model is ... else model, item_set=item_set,
                  normalizer=file_sha256(cut / "vecnormalize.pkl"))
    for k, v in (meta or {}).items():
        if v is ...:
            m.pop(k, None)
        else:
            m[k] = v
    (d / "eval_meta.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return d


def baseline_score(i: int) -> float:
    return (0.0, 0.25, 0.5, 0.0, 1.0, 0.125, 0.0)[i % 7]


def candidate_score(i: int) -> float:
    return min(1.0, baseline_score(i) + (0.5 if i % 5 else 0.0))


def smoke_record(*, base_model: str, item_set: str, **overrides) -> dict[str, Any]:
    """A PASS smoke.json of a CPU-only area (MM-38): no GPU memory, 0 projected GPU-hours, a non-degenerate
    baseline of 20 bounded scores."""
    rec = {"schema": SMOKE_SCHEMA, "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "adapter": ADAPTER, "area": 4, "base_model_hash": base_model, "item_set_hash": item_set,
           "peak_gpu_mem_gb": 0.0, "host_ram_peak_gb": 3.2, "swap_peak_gb": 0.0, "seconds_per_step": 2.4,
           "projected_gpu_h": 0.0, "gpu_log_total_h": 3.12, "baseline_scores": [baseline_score(i) for i in range(20)],
           "error_count": 0, "verdict": "PASS", "reasons": []}
    rec.update(overrides)
    return rec


def guard_log(host_work: Path, label: str, *, lines: list[str] | None = None) -> Path:
    """gpu-step.sh's log: one finished attempt of 105 s (the continue full run's wall time)."""
    host_work.mkdir(parents=True, exist_ok=True)
    lines = lines if lines is not None else [f"2026-11-11T09:00:00Z START {label} attempt=1",
                                             f"2026-11-11T09:01:45Z END {label} attempt=1 rc=0"]
    p = host_work / "gpu-guard.log"
    p.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return p


class Sb3Scenario:
    """A whole area-2 continuation, collected against base cut 2 (module docstring)."""

    def __init__(self, root: Path, *, n_items: int = 300, reg_mutate: Callable[[dict], None] | None = None,
                 overlap: int = 0, base_overlap: int = 0):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.runtime = root / "area2-drones"
        (self.runtime / "env-snapshot" / "shards").mkdir(parents=True)
        shutil.copyfile(SB3 / "env_version.json", self.runtime / "env-snapshot" / "env_version.json")
        shutil.copyfile(SB3 / SHARD, self.runtime / "env-snapshot" / "shards" / SHARD)
        self.data = self.runtime / "env-snapshot" / "shards"
        self.script = self.runtime / "train_ppo.py"
        self.script.write_text(SCRIPT_TEXT, encoding="utf-8")
        self.interpreter = self.runtime / ".venv" / "bin" / "python"
        self.interpreter.parent.mkdir(parents=True)
        self.interpreter.write_text(INTERPRETER_TEXT, encoding="utf-8")
        self.items_dir = root / "items"
        self.item_rows = build_items(self.items_dir, n_items)
        self.items = [r["item"] for r in self.item_rows]
        item_payloads = {r["payload"] for r in self.item_rows}
        (root / "trainer.yaml").write_text("# area 2 trainer config (fixture)\nbackend: sb3_ppo\nprofile: full\n",
                                           encoding="utf-8")
        self.host_work = root / "host-work"
        self.output_root = root / "out"
        gpu_step_script(self.host_work)
        (self.host_work / "gpu-log.md").write_bytes((FIXTURES / "gpu-log.fixture.md").read_bytes())
        rng = random.Random(1000)
        sha = file_sha256(self.script)
        self.base_run = build_run(self.output_root / "area2-drones" / "base-1", phase="base", first=1, last=2,
                                  cuts=(1, 2), seed=1000, base=None, rng=rng, forbidden=item_payloads,
                                  script_sha256=sha)
        self.base = self.base_run / "checkpoint-2"
        self.stage_out = build_run(self.output_root / "area2-drones" / "cont-1", phase="continue", first=3, last=6,
                                   cuts=(4, 6), seed=2000, base=self.base, rng=random.Random(2000),
                                   forbidden=item_payloads, script_sha256=sha)
        if overlap:
            self.inject_overlap(self.stage_out, overlap)
        if base_overlap:
            self.inject_overlap(self.base_run, base_overlap)
        self.final_cut = self.stage_out / "checkpoint-6"
        self.base_model = sb3_model_hash(self.base)
        self.cand_model = sb3_model_hash(self.final_cut)
        self.item_set_hash = manifest_hash(self.items_dir)
        self.reg_mutate = reg_mutate
        self.render()
        self.baseline = build_arm(root / "eval" / "baseline", self.item_rows, baseline_score, cut=self.base,
                                  item_set=self.item_set_hash)
        self.candidate = build_arm(root / "eval" / "candidate", self.item_rows, candidate_score, cut=self.final_cut,
                                   item_set=self.item_set_hash)
        self.launch = write_canonical(root / "launch.json", self.launch_record())
        self.label = self.launch_record()["label"]
        self.smoke = write_canonical(root / "smoke.json", smoke_record(base_model=self.base_model,
                                                                         item_set=self.item_set_hash))
        guard_log(self.host_work, self.label)

    def inject_overlap(self, run: Path, k: int) -> None:
        """Replace the first k episode payloads of the run's first rollout with item payloads (exact overlap)."""
        path = run / "rollouts.jsonl"
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for e, row in zip(lines[0]["episodes"][:k], self.item_rows):
            e["init"], e["payload"] = dict(row["init"]), row["payload"]
        write_jsonl(path, lines)

    def env(self) -> dict[str, str]:
        return {ENV_HOST_WORK: str(self.host_work), ENV_OUTPUT_ROOT: str(self.output_root),
                ENV_INPUT_ROOTS: str(self.root), ENV_INTERPRETER: f"{RUNTIME_CONTAINER}/.venv/bin/python",   # EXT-113
                ENV_INTERPRETER_TREES: str(self.interpreter.parent),
                # the empty host file gatedtrain.testing sets, also for in-process host_config(env())
                "GATEDTRAIN_HOST_CONFIG": os.environ["GATEDTRAIN_HOST_CONFIG"]}

    @property
    def paths(self) -> dict[str, Path]:
        return {"spec": self.root / "spec.yaml", "registration": self.root / "registration.yaml"}

    def render(self, reg_mutate: Callable[[dict], None] | None = None,
               spec_mutate: Callable[[dict], None] | None = None) -> None:
        text = (SB3 / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", self.item_set_hash).replace("{env_version}", ENV_VERSION_SHA)
        text = text.replace("{evaluator_hash}", sha_of({"evaluator": EVALUATOR_ID, "metric": METRIC,
                                                         "direction": "higher_is_better", "scorer": SCORER}))
        mutate = reg_mutate or self.reg_mutate
        if mutate is not None:
            reg = yaml.safe_load(text)
            mutate(reg)
            text = yaml.safe_dump(reg, sort_keys=False)
        (self.root / "registration.yaml").write_text(text, encoding="utf-8")
        spec = (SB3 / "spec.template.yaml").read_text(encoding="utf-8")
        spec = spec.replace("{registration_sha256}", file_sha256(self.root / "registration.yaml")[len("sha256:"):])
        spec = spec.replace("{base}", str(self.base)).replace("{data}", str(self.data))
        if spec_mutate is not None:
            doc = yaml.safe_load(spec)
            spec_mutate(doc)
            spec = yaml.safe_dump(doc, sort_keys=False)
        (self.root / "spec.yaml").write_text(spec, encoding="utf-8")

    def launch_record(self, **overrides) -> dict[str, Any]:
        """ext-launch's record of the continuation (README "Request shapes", continue; gpus false, gpu_est_h 0)."""
        rec = {"schema": RECORD_SCHEMA, "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
               "adapter": ADAPTER, "area": 4, "profile": "full", "label": LABEL, "host_work": str(self.host_work),
               "runtime": {"kind": "venv_in_container", "image": IMAGE_ID, "venv_lock_sha256": "sha256:" + "82" * 32},
               "framework": {"name": "stable-baselines3", "version": "2.9.2", "commit": None},
               "training_seed": 2000, "gpus": False,
               "argv": [f"{RUNTIME_CONTAINER}/.venv/bin/python", f"{RUNTIME_CONTAINER}/train_ppo.py", "--phase",
                        "continue", "--profile", "full", "--out", "/out", "--env-snapshot",
                        f"{RUNTIME_CONTAINER}/env-snapshot", "--base", BASE_CONTAINER],
               "mounts": [{"host": str(self.runtime), "container": RUNTIME_CONTAINER, "mode": "ro"},
                          {"host": str(self.base), "container": BASE_CONTAINER, "mode": "ro"},
                          {"host": str(self.stage_out), "container": "/out", "mode": "rw"}],
               "tmpfs": ["/tmp"], "network": "none", "network_reason": None, "user": run_user(),
               "env_names": list(ENV_NAMES), "gpu_est_h": 0, "gpu_used_h": 3.12,
               "output_dir": str(self.stage_out), "output_root": str(self.output_root),
               "gpu_step_sha256": file_sha256(self.host_work / "bin" / "gpu-step.sh"), "allow_network": False,
               "interpreter_path": f"{RUNTIME_CONTAINER}/.venv/bin/python",
               "interpreter_sha256": file_sha256(self.interpreter),
               "script_path": f"{RUNTIME_CONTAINER}/train_ppo.py", "script_sha256": file_sha256(self.script),
               "host_config_sha256": host_config_sha256(self.host_work, self.output_root, [self.root],   # EXT-117
                                                         interpreter=f"{RUNTIME_CONTAINER}/.venv/bin/python",
                                                         interpreter_trees=[self.interpreter.parent])}
        rec.update(overrides)
        rec["label"] = launch_label(rec)
        return rec

    def rewrite_launch(self, **overrides) -> None:
        rec = self.launch_record(**overrides)
        self.launch.write_bytes(canonical_bytes(rec))
        self.label = rec["label"]
        guard_log(self.host_work, self.label)

    def edit_json(self, path: Path, fn: Callable[[dict], None]) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fn(doc)
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def edit_rollouts(self, run: Path, fn: Callable[[list[dict]], list[dict]]) -> None:
        path = run / "rollouts.jsonl"
        write_jsonl(path, fn([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]))

    def collect(self, runs_root: Path, *, baseline: Path | None = None, candidate: Path | None = None,
                launch: Path | None = None, smoke: Path | None = None, timeout: int = 300):
        args = ["ext-collect", str(self.paths["spec"]), "--adapter", ADAPTER, "--stage-out", str(self.stage_out),
                "--launch-record", str(launch or self.launch), "--smoke-record", str(smoke or self.smoke),
                "--eval-baseline", str(baseline or self.baseline), "--eval-candidate", str(candidate or self.candidate),
                "--runs-root", str(runs_root)]
        return run_cli(args, cwd=self.root, timeout=timeout, env=self.env())
