"""SB3 PPO adapter (area "control": stable-baselines3 2.9.2 PPO on gym-pybullet-drones hover; MULTIMODAL-SPEC.md
section 6 row 4; tests/README.md section 11.19, EXT-98 to EXT-111).

MM-12: a pure-Python class that imports no framework, torch or pickle loader. It reads JSON, the text
`env-freeze.txt` and the member list of `model.zip` (standard library `zipfile`, bytes only), and hashes everything
else. `vecnormalize.pkl` is a pickle (VecNormalize.save, stable_baselines3/common/vec_env/vec_normalize.py:324-332)
and is only ever hashed; `vecnormalize.json` mirrors its statistics.

Run layout, written by the runtime's `runtime/train_ppo.py` (its README.md
"Outputs of `train_ppo.py`" and "What gatedtrain needs", read 2026-11-12):

- two phases, two launches: `--phase base` trains from scratch (`run_args.json` `base_checkpoint` null), `--phase
  continue --base <cut>` loads a base cut and records `base_checkpoint` (the path it loaded, a container path) and
  `base_cut` (that cut's `cut.json`). Iterations are global: a continuation from base cut I starts at I + 1;
- a cut `checkpoint-<I>/` is written into `.checkpoint-<I>.partial/` and renamed: `model.zip` (SB3 save: resume
  state), `policy.pth` (the zip's `policy.pth` member, the weights), `policy_meta.json`, `vecnormalize.pkl`,
  `vecnormalize.json`, `env_version.json`, then `cut.json` last with the SHA-256 of every other file;
- `rollouts.jsonl` (MM-25, MM-27): one line per PPO iteration I, written at the rollout's end before the update that
  uses it: `iteration`, `phase`, `num_timesteps_first`, `num_timesteps_last` (cumulative env steps, 1-based,
  inclusive), `n_envs`, `n_steps`, `env_params_version`, `shard`, `episodes` (every episode with a step in the
  rollout: `{env, episode, seed, init, payload, steps}`), `utc`. Update I uses exactly rollout I; cut I holds the
  weights after update I;
- `env-freeze.txt` (`stable-baselines3==<v>` and `# gym-pybullet-drones commit <sha>`), `env-runtime.json`
  (`script_sha256`, EXT-83).
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from gatedtrain.canonical import entries_hash, file_sha256, manifest_entries, manifest_hash, text_lines
from gatedtrain.errors import Problem
from gatedtrain.ext.registry import EVALUATOR_ROWS, TRAINER_ROWS
from gatedtrain.ext.sdk import (bound_mounts as _bound_mounts, container_file, plain_file as _plain_file, read_run_args,
                        ro_table as _ro_table, runtime_script_problems, sha_or_none as _sha_or_none,
                        within as _within)

NAME = "sb3_ppo"
LIBRARY = "stable-baselines3 2.9.2 PPO with the area runtime's cut callback (train_ppo.py save_cut)"
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")        # train_ppo.py save_cut; `.checkpoint-<I>.partial` never
CUT_MARKER = "cut.json"                                       # written last (train_ppo.py save_cut)
CUT_FILES = ("model.zip", "policy.pth", "policy_meta.json", "vecnormalize.pkl", "vecnormalize.json",
             "env_version.json")                              # train_ppo.py CUT_FILES
COMPONENTS = tuple(TRAINER_ROWS[NAME]["components"])          # EXT-99: the deployable subset (MM-17, MM-19)
NORMALIZER = "vecnormalize.pkl"                               # hashed, never opened (MM-12)
ENV_VERSION = "env_version.json"
ZIP_WEIGHTS = "policy.pth"                                    # the member of model.zip the evaluator loads
FRAMEWORK_PACKAGE = "stable-baselines3"
COMMIT_LINE_RE = re.compile(r"^# gym-pybullet-drones commit [0-9a-f]{40}$")
FREEZE_FILE = "env-freeze.txt"
ROLLOUTS = "rollouts.jsonl"
PHASE_BASE, PHASE_CONTINUE = "base", "continue"

CHECKPOINT_COMPLETENESS = {
    "library": LIBRARY,
    "rule": ("a checkpoint-<I> directory is a cut iff its cut.json, written last after an atomic rename from "
             ".checkpoint-<I>.partial, parses with iteration = I and lists exactly the other files of the directory "
             "(model.zip, policy.pth, policy_meta.json, vecnormalize.pkl, vecnormalize.json, env_version.json), each "
             "with its SHA-256, and model.zip's policy.pth member is the cut's policy.pth; .partial directories are "
             "never cuts"),
    "complete_dir_pattern": CHECKPOINT_DIR_RE.pattern,
    "sources": ["runtime/train_ppo.py save_cut (cut.json last, os.replace)",
                "runtime/README.md \"Cut layout\"",
                "stable-baselines3 2.9.2 common/save_util.py:319-333 (model.zip members)"],
    "verified": True,                                # the layout was read from the sanity runs (README)
}
SEED_DERIVATION = ("PPO(seed) and PPO.load(..., seed=seed) call set_random_seed(seed) (stable_baselines3 2.9.2 "
                   "common/on_policy_algorithm.py:115-117); VecEnv.seed(seed) makes env i reset first with seed + i "
                   "(common/vec_env/base_vec_env.py:308). The training seed is run_args.json `seed`, which must equal "
                   "the launch record's training_seed; 05 section 14 H(run_seed, step, data_ref, sample_index) is not "
                   "claimed.")


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def read_cut_marker(d: Path) -> tuple[dict | None, str | None]:
    """(`<d>/cut.json` as an object, None) or (None, why not)."""
    path = Path(d) / CUT_MARKER
    if not _plain_file(path):
        return None, f"{CUT_MARKER} (written last) is missing"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        return None, f"{CUT_MARKER} does not parse ({exc.__class__.__name__})"
    if not isinstance(doc, dict) or not isinstance(doc.get("files"), dict):
        return None, f"{CUT_MARKER} has no `files` object"
    return doc, None


def read_rollouts(path: Path) -> tuple[list[dict] | None, str | None]:
    """([line objects] in file order, None) or (None, why `rollouts.jsonl` cannot be read)."""
    if not _plain_file(path):
        return None, f"{path} is not a plain file"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"{path.name} cannot be read ({exc.__class__.__name__})"
    out = []
    for n, line in enumerate(text_lines(text), start=1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except (ValueError, RecursionError):
            return None, f"{path.name} line {n} does not parse"
        if not isinstance(doc, dict):
            return None, f"{path.name} line {n} is not a JSON object"
        out.append(doc)
    return out, None


def rollout_problems(lines: list[dict], *, first: int, last: int, rollout_size: int, shard: str,
                     name: str = ROLLOUTS) -> list[str]:
    """EXT-102: the log covers iterations `first` .. `last` exactly, one line each, in order; each line's chunk is
    `rollout_size` env steps, contiguous with the previous line's, its episodes' steps sum to it, and its shard is
    `shard`. Later lines (after the last cut) are allowed: they are orphaned, never consumed (MM-27)."""
    out: list[str] = []
    its = [d.get("iteration") for d in lines]
    want = list(range(first, last + 1))
    if its[:len(want)] != want:
        return [f"{name} iterations {its[:12]}{' ...' if len(its) > 12 else ''} are not {first} .. {last} in order, "
                "one line each (MM-25)"]
    if len(set(map(str, its))) != len(its) or not all(_int(i) for i in its) or its != sorted(its):
        return [f"{name} repeats or reorders iterations after {last}"]
    prev_last = None
    for d in lines[:len(want)]:
        it, lo, hi = d.get("iteration"), d.get("num_timesteps_first"), d.get("num_timesteps_last")
        if not (_int(lo) and _int(hi) and hi - lo + 1 == rollout_size):
            out.append(f"{name} iteration {it}: chunk {lo!r} .. {hi!r} is not {rollout_size} env steps "
                       "(n_steps x n_envs)")
        elif prev_last is not None and lo != prev_last + 1:
            out.append(f"{name} iteration {it}: chunk starts at {lo}, not after the previous chunk's {prev_last}")
        prev_last = hi if _int(hi) else None
        eps = d.get("episodes")
        steps = (sum(e.get("steps") for e in eps) if isinstance(eps, list)
                 and all(isinstance(e, dict) and _int(e.get("steps")) for e in eps) else None)
        if steps != rollout_size:
            out.append(f"{name} iteration {it}: its episodes log {steps!r} env steps, not {rollout_size}")
        if d.get("shard") != shard:
            out.append(f"{name} iteration {it}: shard {d.get('shard')!r} is not {shard!r}")
        if len(out) >= 8:
            break
    return out


class Sb3PpoAdapter:
    """The MM-12 interface for SB3 PPO hover runs (EXT-99 to EXT-108)."""

    name = NAME
    trainer_id = TRAINER_ROWS[NAME]["uses"][0]
    evaluator_ids = tuple(TRAINER_ROWS[NAME]["evaluators"])
    framework_name = FRAMEWORK_PACKAGE
    checkpoint_completeness = CHECKPOINT_COMPLETENESS
    seed_derivation = SEED_DERIVATION
    binds_spec = bool(TRAINER_ROWS[NAME]["spec_binding"])
    uses_env = True                                   # collect passes the registration's eval_distribution (EXT-103)

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    # --- cuts (MM-17, MM-20; EXT-99) ---

    def model_dir_problem(self, d: Path) -> str | None:
        """Review 9 M1 (EXT-154): a file that cannot be read makes the directory incomplete, never an exception."""
        try:
            return self.model_dir_problem_unguarded(d)
        except OSError as exc:
            return (f"{Path(exc.filename).name if exc.filename else Path(d).name} cannot be read "
                    f"({exc.__class__.__name__}: {exc.strerror or exc})")

    def model_dir_problem_unguarded(self, d: Path) -> str | None:
        """None when `d` is a complete cut directory: cut.json lists exactly the other files, every SHA-256 matches,
        and model.zip's policy.pth member is the cut's policy.pth (the evaluator loads the zip)."""
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        doc, why = read_cut_marker(d)
        if doc is None:
            return why
        files = doc["files"]
        if sorted(files) != sorted(CUT_FILES):
            return f"{CUT_MARKER} lists {sorted(files)}, not {sorted(CUT_FILES)}"
        present = sorted(p.name for p in d.iterdir())
        if present != sorted([*CUT_FILES, CUT_MARKER]):
            extra = sorted(set(present) - set(CUT_FILES) - {CUT_MARKER})
            return f"the directory holds {extra or present}, which {CUT_MARKER} does not list"
        for name in CUT_FILES:
            if not _plain_file(d / name):
                return f"{name} is not a plain file"
            if file_sha256(d / name) != files[name]:
                return f"{name} is not the SHA-256 {CUT_MARKER} records"
        try:
            with zipfile.ZipFile(d / "model.zip") as z:
                member = z.read(ZIP_WEIGHTS)
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            return f"model.zip has no readable {ZIP_WEIGHTS} member ({exc.__class__.__name__})"
        if "sha256:" + hashlib.sha256(member).hexdigest() != files["policy.pth"]:
            return f"model.zip's {ZIP_WEIGHTS} member is not the cut's policy.pth"
        return None

    def cut_problem(self, d: Path, step: int) -> str | None:
        bad = self.model_dir_problem(d)
        if bad:
            return bad
        doc, _ = read_cut_marker(d)
        it = doc.get("iteration")
        if not (_int(it) and it == step):
            return f"{CUT_MARKER} iteration is {it!r}, not {step}"
        return None

    def checkpoint_dirs(self, stage_out: Path) -> list[tuple[int, Path]]:
        """Every `checkpoint-<I>` directory, complete or not, in iteration order; `.checkpoint-<I>.partial` is
        never one (EXT-99)."""
        stage_out = Path(stage_out)
        if not stage_out.is_dir():
            return []
        out = []
        for p in stage_out.iterdir():
            m = CHECKPOINT_DIR_RE.match(p.name)
            if m and p.is_dir() and not p.is_symlink():
                out.append((int(m.group(1)), p))
        return sorted(out)

    def find_cuts(self, stage_out: Path) -> tuple[list[tuple[int, Path]], list[Path]]:
        complete, incomplete = [], []
        for step, d in self.checkpoint_dirs(stage_out):
            if self.cut_problem(d, step) is None:
                complete.append((step, d))
            else:
                incomplete.append(d)
        return complete, incomplete

    def identity(self, cut: Path, base: Any = None, *, entries: list[dict] | None = None) -> dict[str, Any]:
        """EXT-99: `model_hash` over the deployable components (weights, policy description, normalizer and its
        JSON mirror, env version); `cut_hash` over the whole cut (model.zip resume state and cut.json too)."""
        if entries is None:
            entries = manifest_entries(Path(cut))
        comps = [e for e in entries if e["path"] in COMPONENTS]
        return {"model_hash": entries_hash(comps), "cut_hash": entries_hash(entries),
                "components": [e["path"] for e in comps], "base_ref": None}

    def normalizer_sha256(self, cut: Path) -> str:
        """MM-19 (EXT-109): the SHA-256 of the cut's VecNormalize file, which is never opened."""
        return file_sha256(Path(cut) / NORMALIZER)

    def env_version_sha256(self, cut: Path) -> str:
        return file_sha256(Path(cut) / ENV_VERSION)

    # --- consumption (MM-25, MM-27; EXT-102) ---

    @staticmethod
    def _shard_name(data: Path) -> str | None:
        names = sorted(p.name for p in Path(data).glob("*.jsonl"))
        return names[0] if len(names) == 1 else None

    def consumption_problems(self, stage_out: Path, data: Path, cuts: list) -> list[str]:
        """EXT-102: why the observed log cannot give the consumption of iterations first_iteration .. the last cut
        (refused before writing)."""
        args, why = read_run_args(stage_out)
        if args is None:
            return [why]
        if not cuts:
            return ["no complete cut"]
        shard = self._shard_name(data)
        if shard is None:
            return [f"the spec's env-params snapshot {data} holds {len(list(Path(data).glob('*.jsonl')))} shards, not "
                    "one per env_params_version (C3)"]
        s = args.get("settings") if isinstance(args.get("settings"), dict) else {}
        n_envs, n_steps, first = s.get("n_envs"), s.get("n_steps"), args.get("first_iteration")
        if not (_int(n_envs) and _int(n_steps) and n_envs > 0 and n_steps > 0 and _int(first) and first >= 1):
            return ["run_args.json has no positive integer settings.n_envs, settings.n_steps and first_iteration"]
        lines, why = read_rollouts(Path(stage_out) / ROLLOUTS)
        if lines is None:
            return [why]
        last = max(step for step, _ in cuts)
        return rollout_problems(lines, first=first, last=last, rollout_size=n_envs * n_steps, shard=shard)

    def consumption(self, stage_out: Path, data: Path, cuts: list, *, run_id: str = "",
                    attempt_id: str = "0") -> dict[str, Any]:
        """Precision `iteration`, derivation `observed` (chosen 2026-11-12): one ledger row per
        (iteration, shard) for this run's iterations first_iteration .. the last cut, read from rollouts.jsonl as
        the run wrote it. `data_ref.row`, `chunk` and `seed_inputs` are null as 05 section 12.2 fixes for an
        `iteration` record; the per-episode lines stay in the log (EXT-107). The base run's rollouts are the base
        run's consumption, never this run's (EXT-102)."""
        from gatedtrain.ledger import WRITER_ID, check_precision

        bad = self.consumption_problems(stage_out, data, cuts)
        if bad:                                          # collect refuses these before writing
            raise ValueError("; ".join(bad))
        check_precision("iteration", "observed")
        args, _ = read_run_args(stage_out)
        first, last = args["first_iteration"], max(step for step, _ in cuts)
        shard = self._shard_name(data)
        snapshot = manifest_hash(data)
        rows = []
        for step in range(first, last + 1):
            sample_id = hashlib.sha256(f"{run_id}\x00{attempt_id}\x00{step}\x00{shard}".encode()).hexdigest()
            rows.append({
                "run_id": run_id, "attempt_id": attempt_id, "step": step, "sample_id": "sha256:" + sample_id,
                "data_ref": {"snapshot": snapshot, "shard": shard, "row": None, "mixture": None, "component": None},
                "parent_ref": None, "outcome": "consumed", "reason": None, "rule_version": None,
                "gen_version": None, "reward_epoch": None, "rewards": None, "sampler_config_hash": None,
                "seed_inputs": None, "chunk": None, "precision": "iteration", "derivation": "observed",
                "writer_id": WRITER_ID,
            })
        return {"precision": "iteration", "rows": rows, "derivation": "observed",
                "reason": (f"{ROLLOUTS} logs every rollout at its end, before the update that uses it (MM-25, MM-27): "
                           f"iterations {first} .. {last} of this run each read shard {shard}; derivation observed; "
                           "the base run's iterations are the base run's consumption, not this run's")}

    # --- exact overlap (MM-11; EXT-107) ---

    def training_payloads(self, stage_out: Path, base: Path, cuts: list, *,
                          data: Path | None = None) -> tuple[set[str] | None, str | None]:
        """Every episode payload the candidate's weights were trained on: this run's rollouts up to its last cut and
        the base run's (the base cut's parent directory) up to the base cut, both phases. (None, why) when a log
        cannot be read. `data` (the env-params shard) is not read: it holds parameter ranges, not states (EXT-107)."""
        base_args, why = read_run_args(Path(base).parent)
        if base_args is None:
            return None, f"the base run {Path(base).parent}: {why}"
        doc, why = read_cut_marker(base)
        if doc is None or not _int(doc.get("iteration")):
            return None, f"the base cut {base}: {why or 'cut.json has no iteration'}"
        out: set[str] = set()
        for where, upto in ((Path(base).parent, doc["iteration"]), (Path(stage_out), max(s for s, _ in cuts))):
            lines, why = read_rollouts(where / ROLLOUTS)
            if lines is None:
                return None, f"{where}: {why}"
            for d in lines:
                if not (_int(d.get("iteration")) and d["iteration"] <= upto):
                    continue
                for e in d.get("episodes") or []:
                    if isinstance(e, dict) and isinstance(e.get("payload"), str):
                        out.add(e["payload"])
        return out, None

    # --- pins and checks (MM-33, MM-21; EXT-101, EXT-103, EXT-108) ---

    def framework_identity(self, stage_out: Path, launch: dict) -> dict[str, Any]:
        """`stable-baselines3==<v>` in env-freeze.txt equals the launch record's framework, and the freeze names the
        gym-pybullet-drones commit (MM-33). Raises ValueError with the reason."""
        freeze = Path(stage_out) / FREEZE_FILE
        if not _plain_file(freeze):
            raise ValueError(f"{freeze} is missing")
        versions: dict[str, str] = {}
        lines = [l.strip() for l in text_lines(freeze.read_text(encoding="utf-8"))]
        for line in lines:
            if "==" in line and not line.startswith("#"):
                name, _, version = line.partition("==")
                versions[name.strip().lower().replace("_", "-")] = version.strip()
        version = versions.get(FRAMEWORK_PACKAGE)
        if version is None:
            raise ValueError(f"{freeze} has no `{FRAMEWORK_PACKAGE}==<version>` line")
        if not any(COMMIT_LINE_RE.fullmatch(l) for l in lines):
            raise ValueError(f"{freeze} has no `# gym-pybullet-drones commit <40 hex>` line (the source checkout's pin)")
        fw = launch.get("framework") if isinstance(launch.get("framework"), dict) else {}
        if fw.get("name") != FRAMEWORK_PACKAGE or fw.get("version") != version:
            raise ValueError(f"{freeze} pins {FRAMEWORK_PACKAGE}=={version}, but the launch record claims "
                             f"framework {fw.get('name')!r} {fw.get('version')!r}")
        return {"name": FRAMEWORK_PACKAGE, "version": version, "freeze_sha256": file_sha256(freeze)}

    def script_problems(self, stage_out: Path, launch: dict) -> list[Problem]:
        """EXT-83: env-runtime.json `script_sha256` equals the record's (train_ppo.py writes it as area 1 does)."""
        return runtime_script_problems(stage_out, launch)

    BASE_FIX = ("continue from the spec's base cut mounted read-only (e.g. --base /inputs/base, recorded in "
                "run_args.json) and record that mount in the launch record")
    DATA_FIX = ("train with the spec's env-params shard in a read-only mount (recorded in run_args.json "
                "env_params_file and env_params_sha256)")

    def mount_binding_problems(self, mounts: Any, *, base: Path, data: Path, base_field: str, data_field: str,
                               flag: str = "--launch-record") -> list[Problem]:
        """`ext-launch --spec` (optional for this adapter): a read-only mount whose host path is the spec's base cut,
        and the spec's env-params directory inside a read-only mount (EXT-101, EXT-103)."""
        out = []
        if not _bound_mounts(mounts, base):
            out.append(Problem("EXT-BASE", base_field, f"{flag} mounts no read-only directory whose host path is the "
                               f"spec's base cut {base}", self.BASE_FIX))
        want = str(Path(data).resolve())
        if not any(_within(want, host) for _, host in _ro_table(mounts)):
            out.append(Problem("EXT-DATA", data_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's env-params directory {data} (EXT-103)", self.DATA_FIX))
        return out

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str, stage_out: Path | None = None,
               data: Path | None = None, data_field: str = "data",
               eval_distribution: dict | None = None) -> list[Problem]:
        """EXT-101 base binding, EXT-103 data and environment binding, EXT-108 seed, profile and last cut."""
        stage_out = Path(stage_out) if stage_out is not None else Path(last_cut).parent
        out: list[Problem] = []
        args, why = read_run_args(stage_out)
        if args is None:
            return [Problem("EXT-BASE", base_field, f"{why}, so the run cannot be bound to the spec's base cut "
                            "(EXT-101)", self.BASE_FIX)]
        base_doc, why = read_cut_marker(base)
        # EXT-101: the continuation loaded the spec's base cut, by its read-only mount, and recorded its cut.json.
        if args.get("phase") != PHASE_CONTINUE:
            out.append(Problem("EXT-BASE", base_field, f"run_args.json phase is {args.get('phase')!r}: ext-collect gates "
                               "a continue-phase run against the base cut it loaded; a base-phase run trains from "
                               "scratch and is the baseline's producer, not a candidate (EXT-101)",
                               "collect the continue-phase run whose --base is the spec's base cut"))
        mounts = _bound_mounts(launch.get("mounts"), base)
        want = sorted({m["container"] for m in mounts if isinstance(m.get("container"), str)})
        got = args.get("base_checkpoint")
        if not mounts:
            out.append(Problem("EXT-BASE", base_field, f"the launch record mounts no read-only directory whose host "
                               f"path is the spec's base cut {base} (EXT-101)", self.BASE_FIX))
        elif got not in want:
            out.append(Problem("EXT-BASE", base_field, f"run_args.json base_checkpoint is {got!r}, not the base mount's "
                               f"container path {want} (EXT-101)", self.BASE_FIX))
        recorded = args.get("base_cut")
        if base_doc is not None:
            if not (isinstance(recorded, dict) and recorded.get("files") == base_doc["files"]
                    and recorded.get("iteration") == base_doc.get("iteration")):
                out.append(Problem("EXT-BASE", base_field, "run_args.json base_cut (the cut.json the run loaded) is not "
                                   f"the spec's base cut's {base / CUT_MARKER}: the run started from other weights "
                                   "(EXT-101)", self.BASE_FIX))
            it = base_doc.get("iteration")
            if not (_int(it) and args.get("first_iteration") == it + 1):
                out.append(Problem("EXT-BASE", base_field, f"run_args.json first_iteration is "
                                   f"{args.get('first_iteration')!r}, not the base cut's iteration {it!r} + 1 (EXT-101)",
                                   self.BASE_FIX))
        base_args, why = read_run_args(Path(base).parent)
        if base_args is None:
            out.append(Problem("EXT-BASE", base_field, f"the base run {Path(base).parent}: {why}; the base cut must lie "
                               "in its base-phase run's output directory (EXT-101)",
                               "point the spec's base at checkpoint-<I> inside the base run's output directory"))
        elif base_args.get("phase") != PHASE_BASE or base_args.get("base_checkpoint") is not None:
            out.append(Problem("EXT-BASE", base_field, f"the base run {Path(base).parent} is phase "
                               f"{base_args.get('phase')!r} with base_checkpoint {base_args.get('base_checkpoint')!r}, "
                               "not a base-phase run trained from scratch: chains of more than two phases are not "
                               "supported (EXT-101)", "use a cut of the base-phase run as the spec's base"))
        # EXT-103: the env-params shard the run read is the spec's, through a read-only mount; the env version is the
        # registered evaluation environment's.
        if data is not None:
            shard = self._shard_name(data)
            path = args.get("env_params_file")
            hit = container_file(path, launch.get("mounts")) if isinstance(path, str) else None
            if shard is None or hit is None or Path(hit[1]).resolve() != (Path(data) / shard).resolve():
                out.append(Problem("EXT-DATA", data_field, f"run_args.json env_params_file {path!r} is not, through a "
                                   f"read-only mount, the spec's env-params shard {Path(data) / (shard or '?')} "
                                   "(EXT-103)", self.DATA_FIX))
            elif args.get("env_params_sha256") != _sha_or_none(Path(data) / shard):
                out.append(Problem("EXT-DATA", data_field, f"run_args.json env_params_sha256 "
                                   f"{args.get('env_params_sha256')!r} is not the SHA-256 of {Path(data) / shard} "
                                   "(EXT-103)", self.DATA_FIX))
        dist = eval_distribution if isinstance(eval_distribution, dict) else {}
        env = dist.get("env_version")
        if env is not None:
            pairs = [("run_args.json env_version_sha256", args.get("env_version_sha256")),
                     (f"{Path(last_cut).name}/{ENV_VERSION}", self.env_version_sha256(last_cut)),
                     (f"the base cut's {ENV_VERSION}", self.env_version_sha256(base) if Path(base, ENV_VERSION).is_file()
                      else None)]
            for what, value in pairs:
                if value != env:
                    out.append(Problem("EXT-ENV", "registration.eval_distribution.env_version",
                                       f"{what} is {value!r}, not the registered evaluation environment {env} (MM-19, "
                                       "EXT-103)", "train and evaluate in the registered environment version"))
        version = dist.get("env_params_version")
        if version is not None and args.get("env_params_version") != version:
            out.append(Problem("EXT-ENV", "registration.eval_distribution.env_params_version",
                               f"run_args.json env_params_version is {args.get('env_params_version')!r}, not the "
                               f"registered {version!r} (EXT-103)", "train on the registered env_params_version"))
        # EXT-108: seed, profile, and the run finished the iterations it was launched for.
        seed, claimed = args.get("seed"), launch.get("training_seed")
        if not (_int(seed) and seed == claimed):
            out.append(Problem("EXT-LAUNCH", "--launch-record.training_seed",
                               f"the launch record claims training_seed {claimed!r}, but run_args.json `seed` is "
                               f"{seed!r} (MM-29, EXT-108)", "launch with the seed the launch record names"))
        if args.get("profile") != launch.get("profile"):
            out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"run_args.json `profile` is "
                               f"{args.get('profile')!r}, the launch record's {launch.get('profile')!r} (EXT-108)",
                               "collect the run launched with this record's profile"))
        m = CHECKPOINT_DIR_RE.match(Path(last_cut).name)
        last = int(m.group(1)) if m else None
        if not (_int(args.get("last_iteration")) and args["last_iteration"] == last):
            out.append(Problem("EXT-STAGE", "--stage-out", f"the last cut is iteration {last}, but run_args.json "
                               f"last_iteration is {args.get('last_iteration')!r}: the run did not save its last "
                               "iteration (EXT-108)", "collect a run that finished and saved its last iteration"))
        return out
