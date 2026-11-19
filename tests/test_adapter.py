"""External-run adapters, slice 2: area "control" (SB3 PPO hover; tests/README.md section 11.19, EXT-98 to EXT-111).

Sources: MULTIMODAL-SPEC.md MM-1 to MM-39 (section 6 row 4 and its accepted deviations), design 05 section 11
(lineage, impact across runs through `used` model edges), section 12.1 (environment parameters as a DatasetSnapshot,
R11-17, C3), section 12.2 (an `iteration` record has null row, chunk and seed inputs), section 11.2 (derivation,
R11-18, C4), 05 section 3.3 (an item is an episode seed with environment parameters, R11-19, C5); the area runtime
`runtime/README.md` ("What gatedtrain needs", read 2026-11-12). Helpers:
`tests/_sb3.py`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from gatedtrain.testing.ext import ENV_INPUT_ROOTS, IMAGE_ID, sha_of, write_canonical
from _sb3 import (ADAPTER, BASE_CONTAINER, ENV_NAMES, ENV_VERSION_SHA, EVAL_HOVER, EVAL_HOVER_SHA256, EVALUATOR_ID,
                  HAS_TRL, NO_TRL, RUNTIME_CONTAINER, SAMPLER, SB3, SCORER, SHARD, SUPPORT_ID, TRAINER_ID, TRL_ADAPTER,
                  TRL_EVALUATOR_ID, Sb3Scenario, baseline_score, build_arm, build_cut, guard_log, sb3_model_hash,
                  run_user, smoke_record)
from gatedtrain.testing.support import (EXIT_OK, EXIT_REFUSED, describe, file_sha256, manifest_hash, parse_errors,
                                        parse_run_dir, run_cli)


def registry():
    from gatedtrain.ext import registry as reg
    return reg


def adapter():
    return registry().adapter(ADAPTER)


class Sb3Case(unittest.TestCase):
    """One scenario per class; each test that changes it restores it (or builds its own)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-sb3-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = Sb3Scenario(cls.tmp / "sc")
        cls.n = 0

    def fresh(self, **kw) -> Sb3Scenario:
        type(self).n += 1
        return Sb3Scenario(self.tmp / f"sc-{type(self).n}", **kw)

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def passes(self, sc: Sb3Scenario | None = None, **kw) -> Path:
        proc = (sc or self.sc).collect(self.runs(), **kw)
        self.assertIn(proc.returncode, (EXIT_OK, 1), describe(proc))
        return parse_run_dir(proc.stdout)

    def refused(self, rule: str, field_prefix: str | None = None, *, sc: Sb3Scenario | None = None,
                text: str | None = None, **kw):
        runs = self.runs()
        proc = (sc or self.sc).collect(runs, **kw)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))
                and (text is None or text in e["message"])]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix}... {text!r}; got {errs}\n{describe(proc)}")
        self.assertFalse(runs.exists() and any(runs.iterdir()), "a refusal writes no run directory")
        return hits


# ---------------------------------------------------------------------------
# EXT-98: registry rows, item set shape
# ---------------------------------------------------------------------------

class TestRegistryRows(unittest.TestCase):

    def test_trainer_row(self):
        row = registry().TRAINER_ROWS[ADAPTER]
        self.assertEqual(list(row["uses"]), [TRAINER_ID])
        self.assertEqual(list(row["evaluators"]), [EVALUATOR_ID, SUPPORT_ID])        # deciding first
        self.assertEqual(row["consumption"], {"target": "iteration", "fallback": None})
        self.assertEqual(set(row["pins"]), {"stable-baselines3", "gymnasium", "gym-pybullet-drones", "pybullet"})
        self.assertEqual(row["framework"], "stable-baselines3")
        self.assertEqual(tuple(row["components"]), ("policy.pth", "policy_meta.json", "vecnormalize.pkl",
                                                    "vecnormalize.json", "env_version.json"))

    def test_evaluator_rows(self):
        rows = registry().EVALUATOR_ROWS
        for eid, metric, tier in ((EVALUATOR_ID, "hover_success", "T3"), (SUPPORT_ID, "hover_score", "T0")):
            with self.subTest(evaluator=eid):
                row = rows[eid]
                self.assertEqual((row["metric"], row["direction"], tuple(row["score_range"]), row["tier"]),
                                 (metric, "higher_is_better", (0, 1), tier))
                self.assertFalse(row["binary"])
                self.assertEqual(tuple(row["error_classes"]), ("simulator_error",))
                self.assertEqual(set(row["score_outcomes"]), {"crash", "below_tolerance", "timeout"})
                self.assertEqual(row["payload_text_field"], "payload")
                self.assertEqual(tuple(row["item_payload"]), ("init", "payload"))
                self.assertTrue(row["normalizer"] and row["env_version"])
                self.assertEqual(set(row["sampler_keys"]), set(SAMPLER))
                self.assertEqual(row["scorer"]["code_sha256"], EVAL_HOVER_SHA256)
                self.assertEqual(row["scorer"]["name"], metric)
        self.assertEqual(rows[EVALUATOR_ID]["scorer"], SCORER)

    def test_identity_hash(self):
        """EXT-6 for the new evaluator: SHA-256 of canonical {evaluator, metric, direction, scorer}."""
        self.assertEqual(registry().evaluator_identity_hash(EVALUATOR_ID),
                         sha_of({"evaluator": EVALUATOR_ID, "metric": "hover_success",
                                 "direction": "higher_is_better", "scorer": SCORER}))

    @unittest.skipUnless(EVAL_HOVER.is_file(), "the area-2 runtime is absent")
    def test_scorer_code_hash_is_eval_hover(self):
        self.assertEqual(file_sha256(EVAL_HOVER), EVAL_HOVER_SHA256)

    def test_item_set_init_is_an_object(self):
        """EXT-98: `init` is a JSON object, `payload` a string; a string `init` or a missing payload is refused."""
        from gatedtrain.ext.collect import read_item_set
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-sb3-items-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        row = registry().EVALUATOR_ROWS[EVALUATOR_ID]
        args = (tuple(row["item_payload"]), tuple(row["item_payload_objects"]))
        good = {"item": "a", "env_params_version": "hover-init-v1", "init": {"x": 0.1, "y": 0, "z": 1, "vz": 0},
                "payload": "hover-init-v1 x=0.1000 y=0.0000 z=1.0000 vz=0.0000"}
        (tmp / "items.jsonl").write_text(json.dumps(good) + "\n", encoding="utf-8")
        self.assertEqual(read_item_set(tmp, *args)[1], [])
        for bad in ({**good, "init": "x=0.1"}, {k: v for k, v in good.items() if k != "payload"},
                    {**good, "payload": {"x": 1}}):
            with self.subTest(bad=bad):
                (tmp / "items.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
                self.assertTrue(read_item_set(tmp, *args)[1])


# ---------------------------------------------------------------------------
# EXT-99: cuts, completeness and identity
# ---------------------------------------------------------------------------

class TestCuts(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-sb3-cuts-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = self.tmp / "run"
        for it in (2, 4, 6):
            build_cut(self.out, it, phase="continue")

    def found(self):
        complete, incomplete = adapter().find_cuts(self.out)
        return [s for s, _ in complete], sorted(Path(p).name for p in incomplete)

    def test_complete(self):
        self.assertEqual(self.found(), ([2, 4, 6], []))

    def test_partial_directory_ignored(self):
        """`.checkpoint-<I>.partial` (save_cut writes there, then renames) is never a cut."""
        shutil.copytree(self.out / "checkpoint-6", self.out / ".checkpoint-8.partial")
        self.assertEqual(self.found(), ([2, 4, 6], []))

    def test_incomplete_cuts(self):
        def no_marker(d): (d / "cut.json").unlink()
        def changed(d): (d / "vecnormalize.pkl").write_bytes(b"other")
        def wrong_iteration(d):
            doc = json.loads((d / "cut.json").read_text()); doc["iteration"] = 5
            (d / "cut.json").write_text(json.dumps(doc))
        def extra(d): (d / "notes.txt").write_text("x")
        def missing(d): (d / "env_version.json").unlink()
        def zip_differs(d):
            with zipfile.ZipFile(d / "model.zip", "w") as z:
                z.writestr("policy.pth", b"other weights")
            doc = json.loads((d / "cut.json").read_text()); doc["files"]["model.zip"] = file_sha256(d / "model.zip")
            (d / "cut.json").write_text(json.dumps(doc))
        def unlisted(d):
            doc = json.loads((d / "cut.json").read_text()); del doc["files"]["vecnormalize.json"]
            (d / "cut.json").write_text(json.dumps(doc))
        for fn in (no_marker, changed, wrong_iteration, extra, missing, zip_differs, unlisted):
            with self.subTest(case=fn.__name__):
                shutil.rmtree(self.out / "checkpoint-4")
                build_cut(self.out, 4, phase="continue")
                fn(self.out / "checkpoint-4")
                self.assertEqual(self.found(), ([2, 6], ["checkpoint-4"]))

    def test_identity(self):
        cut = self.out / "checkpoint-6"
        ident = adapter().identity(cut)
        self.assertEqual(ident["model_hash"], sb3_model_hash(cut))
        self.assertEqual(ident["cut_hash"], manifest_hash(cut))
        self.assertEqual(sorted(ident["components"]), sorted(registry().TRAINER_ROWS[ADAPTER]["components"]))
        self.assertIsNone(ident["base_ref"])
        before = ident["model_hash"]
        with zipfile.ZipFile(cut / "model.zip", "a") as z:          # resume state only: model_hash unchanged
            z.writestr("system_info.txt", "other host")
        self.assertEqual(adapter().identity(cut)["model_hash"], before)
        (cut / "vecnormalize.pkl").write_bytes(b"other statistics")  # the normalizer is a component (MM-19)
        self.assertNotEqual(adapter().identity(cut)["model_hash"], before)

    def test_pickle_never_opened(self):
        """MM-12: the normalizer is hashed, never unpickled; a file that would run code when unpickled is only hashed."""
        cut = self.out / "checkpoint-6"
        (cut / "vecnormalize.pkl").write_bytes(b"cos\nsystem\n(S'touch /nonexistent/x'\ntR.")
        self.assertEqual(adapter().normalizer_sha256(cut), file_sha256(cut / "vecnormalize.pkl"))


# ---------------------------------------------------------------------------
# EXT-111: the whole continuation collected (and EXT-101, EXT-102, EXT-105 outputs)
# ---------------------------------------------------------------------------

class TestCollectPass(Sb3Case):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        proc = cls.sc.collect(cls.tmp / "runs-pass")
        cls.proc = proc
        cls.run_dir = parse_run_dir(proc.stdout) if proc.returncode == EXIT_OK else None

    def setUp(self):
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))

    def report(self) -> dict:
        return json.loads((self.run_dir / "report" / "gate_report.json").read_text(encoding="utf-8"))

    def test_pass_and_hashes(self):
        rep = self.report()
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["adapter"], ADAPTER)
        self.assertEqual(rep["independence_tier"], "T3")
        self.assertEqual(rep["baseline_model_hash"], self.sc.base_model)        # the base cut's model hash
        self.assertEqual(rep["candidate_model_hash"], self.sc.cand_model)
        self.assertEqual(rep["contamination"], {"exact_overlap": 0, "allowed": False})
        self.assertEqual(rep["consumption_precision"], "iteration")
        self.assertEqual(rep["n_items"], 300)

    def test_cpu_accounting(self):
        """EXT-105: gpus false: 0 GPU-seconds; the wrapper's START/END time is wall_seconds."""
        rep = self.report()
        lock = json.loads((self.run_dir / "lockfile.json").read_text(encoding="utf-8"))
        self.assertEqual((rep["gpu_seconds"], rep["wall_seconds"]), (0, 105))
        self.assertEqual((lock["gpu_seconds"], lock["wall_seconds"]), (0, 105))
        text = (self.run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn("gpu_seconds: 0\nwall_seconds: 105\n", text)

    def test_ledger_is_this_runs_iterations(self):
        """EXT-102: one observed `iteration` row per iteration 3 .. 6 of the continuation (never the base run's 1 .. 2),
        shard hover-init-v1.jsonl of the env-params snapshot, row, chunk and seed inputs null (05 section 12.2)."""
        import pyarrow.parquet as pq
        rows = pq.read_table(self.run_dir / "ledger" / "consumption.parquet").to_pylist()
        self.assertEqual([r["step"] for r in rows], [3, 4, 5, 6])
        for r in rows:
            self.assertEqual((r["precision"], r["derivation"], r["outcome"]), ("iteration", "observed", "consumed"))
            self.assertEqual(r["data_ref"]["shard"], SHARD)
            self.assertEqual(r["data_ref"]["snapshot"], manifest_hash(self.sc.data))
            self.assertIsNone(r["data_ref"]["row"])
            self.assertIsNone(r["chunk"])
            self.assertIsNone(r["seed_inputs"])

    def test_lineage_binds_the_base_cut(self):
        """EXT-101: the training stage `used` the base cut (cut hash and model hash); the gate's baseline is it."""
        edges = json.loads((self.run_dir / "lineage" / "edges.json").read_text(encoding="utf-8"))
        train, gate = edges["activities"]
        model = [u for u in train["used"] if u["role"] == "model"][0]
        self.assertEqual((model["hash"], model["model_hash"]), (manifest_hash(self.sc.base), self.sc.base_model))
        base = [u for u in gate["used"] if u["role"] == "baseline"][0]
        self.assertEqual(base["hash"], self.sc.base_model)
        self.assertEqual([g["last_step_included"] for g in train["generated"]], [4, 6])

    def test_impact_on_the_shard(self):
        """EXT-102: the shard query returns the continuation's cuts 4 and 6 (first step 3, iteration, observed); the
        base cut, a `used` input whose consumption is the base run's, is not returned from this run directory."""
        proc = run_cli(["impact", str(self.sc.data / SHARD), "--run-dir", str(self.run_dir), "--json"],
                       cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        rows = json.loads(proc.stdout)
        rows = rows["results"] if isinstance(rows, dict) else rows
        self.assertEqual([(r["last_step_included"], r["first_step"], r["precision"], r["derivation"]) for r in rows],
                         [(4, 3, "iteration", "observed"), (6, 3, "iteration", "observed")])
        self.assertNotIn(self.sc.base_model, [r.get("model_hash") for r in rows])

    def test_arm_normalizer_is_the_cut(self):
        """EXT-109: each arm's normalizer is its evaluated cut's; the env version is the registered one."""
        for arm, cut in (("baseline", self.sc.base), ("candidate", self.sc.final_cut)):
            meta = json.loads((self.run_dir / "eval" / arm / "eval_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["normalizer_sha256"], file_sha256(cut / "vecnormalize.pkl"))
            self.assertEqual(meta["env_version_sha256"], ENV_VERSION_SHA)


# ---------------------------------------------------------------------------
# EXT-101: base binding (two phases, two launches)
# ---------------------------------------------------------------------------

class TestBaseBinding(Sb3Case):

    def edit_args(self, sc, fn):
        sc.edit_json(sc.stage_out / "run_args.json", fn)

    def test_base_checkpoint_not_the_mount(self):
        sc = self.fresh()
        self.edit_args(sc, lambda a: a.update(base_checkpoint="/inputs/other"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="base_checkpoint")

    def test_base_cut_files_differ(self):
        sc = self.fresh()
        self.edit_args(sc, lambda a: a["base_cut"]["files"].update({"policy.pth": "sha256:" + "0" * 64}))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="base_cut")

    def test_first_iteration_not_after_the_base(self):
        sc = self.fresh()
        self.edit_args(sc, lambda a: a.update(first_iteration=1))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="first_iteration")

    def test_base_mount_missing(self):
        sc = self.fresh()
        sc.rewrite_launch(mounts=[m for m in sc.launch_record()["mounts"] if m["container"] != BASE_CONTAINER])
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="mounts no read-only directory")

    def test_base_phase_run_is_not_a_candidate(self):
        sc = self.fresh()
        self.edit_args(sc, lambda a: a.update(phase="base"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="phase")

    def test_base_run_must_be_a_base_phase(self):
        sc = self.fresh()
        sc.edit_json(sc.base_run / "run_args.json", lambda a: a.update(phase="continue"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="chains of more than two phases")


# ---------------------------------------------------------------------------
# EXT-102: the observed consumption log
# ---------------------------------------------------------------------------

class TestConsumptionLog(Sb3Case):

    def test_refusals(self):
        def gap(lines): return [l for l in lines if l["iteration"] != 4]
        def size(lines):                                   # the last line: no later chunk to misalign
            lines[-1]["num_timesteps_last"] += 1
            return lines
        def episodes(lines):
            lines[0]["episodes"][0]["steps"] -= 1
            return lines
        def shard(lines):
            lines[2]["shard"] = "other.jsonl"
            return lines
        def jump(lines):
            for l in lines[2:]:
                l["num_timesteps_first"] += 16
                l["num_timesteps_last"] += 16
            return lines
        for fn, text in ((gap, "iterations"), (size, "is not 16 env steps"), (episodes, "episodes log"), (shard, "shard"),
                         (jump, "not after the previous chunk")):
            with self.subTest(case=fn.__name__):
                sc = self.fresh()
                sc.edit_rollouts(sc.stage_out, fn)
                self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text=text)

    def test_orphaned_lines_allowed(self):
        """A line after the last cut (a rollout logged, its cut never written) is orphaned, not consumed (MM-27)."""
        sc = self.fresh()
        def more(lines):
            extra = json.loads(json.dumps(lines[-1]))
            extra.update(iteration=7, num_timesteps_first=lines[-1]["num_timesteps_last"] + 1,
                         num_timesteps_last=lines[-1]["num_timesteps_last"] + 16)
            return lines + [extra]
        sc.edit_rollouts(sc.stage_out, more)
        import pyarrow.parquet as pq
        rows = pq.read_table(self.passes(sc) / "ledger" / "consumption.parquet").to_pylist()
        self.assertEqual([r["step"] for r in rows], [3, 4, 5, 6])

    def test_missing_log_refused(self):
        sc = self.fresh()
        (sc.stage_out / "rollouts.jsonl").unlink()
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc)


# ---------------------------------------------------------------------------
# EXT-103: env-params data binding and the registered environment version
# ---------------------------------------------------------------------------

class TestDataBinding(Sb3Case):

    def test_params_file_outside_the_mounts(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(env_params_file="/elsewhere/hover-init-v1.jsonl"))
        self.refused("EXT-DATA", "artifacts.env_params.uri", sc=sc)

    def test_params_hash_differs(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(env_params_sha256="sha256:" + "1" * 64))
        self.refused("EXT-DATA", "artifacts.env_params.uri", sc=sc, text="env_params_sha256")

    def test_env_version_not_registered(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(env_version_sha256="sha256:" + "2" * 64))
        self.refused("EXT-ENV", "registration.eval_distribution.env_version", sc=sc)

    def test_env_params_version_not_registered(self):
        sc = self.fresh(reg_mutate=lambda r: r["eval_distribution"].update(env_params_version="hover-init-v2"))
        self.refused("EXT-ENV", "registration.eval_distribution.env_params_version", sc=sc)

    def test_registration_without_env_version(self):
        sc = self.fresh(reg_mutate=lambda r: r["eval_distribution"].update(env_version=None))
        self.refused("REG-FIELD", "registration.eval_distribution.env_version", sc=sc)


# ---------------------------------------------------------------------------
# EXT-106: the registered sampler of a deterministic policy
# ---------------------------------------------------------------------------

class TestRegistrationSampler(unittest.TestCase):

    def check(self, sampler) -> list:
        from gatedtrain.registration import check_contents
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-sb3-reg-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        text = (SB3 / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", "sha256:" + "a" * 64).replace("{env_version}", ENV_VERSION_SHA)
        text = text.replace("{evaluator_hash}", "sha256:" + "b" * 64)
        reg = yaml.safe_load(text)
        reg["sampler"] = sampler
        return [p.field for p in check_contents(reg, evaluator_id=EVALUATOR_ID)]

    def test_policy_sampler_accepted(self):
        self.assertEqual([f for f in self.check(dict(SAMPLER)) if f.startswith("registration.sampler")], [])

    def test_refusals(self):
        llm = {"name": "deployment", "temperature": 0.7, "max_new_tokens": 1024}
        self.assertIn("registration.sampler.temperature", self.check(llm))
        self.assertIn("registration.sampler.policy", self.check(llm))
        missing = {k: v for k, v in SAMPLER.items() if k != "max_steps"}
        self.assertIn("registration.sampler.max_steps", self.check(missing))
        self.assertIn("registration.sampler.seed", self.check({**SAMPLER, "seed": 1}))

# ---------------------------------------------------------------------------
# EXT-107: exact overlap against the training episodes, both phases
# ---------------------------------------------------------------------------

class TestOverlap(Sb3Case):

    def test_continuation_overlap_refused(self):
        sc = self.fresh(overlap=2)
        self.refused("EXT-CONTAMINATION", "registration.contamination.allow_exact_overlap", sc=sc,
                     text="2 item(s)")

    def test_base_run_overlap_refused(self):
        sc = self.fresh(base_overlap=1)
        self.refused("EXT-CONTAMINATION", "registration.contamination.allow_exact_overlap", sc=sc,
                     text="1 item(s)")

    def test_allowed_overlap_reported(self):
        sc = self.fresh(overlap=2, base_overlap=1,
                        reg_mutate=lambda r: r["contamination"].update(allow_exact_overlap=True))
        rep = json.loads((self.passes(sc) / "report" / "gate_report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["contamination"], {"exact_overlap": 2, "allowed": True})   # the same items in both runs

    def test_not_against_the_shard(self):
        """The shard holds parameter ranges, not states: an item whose payload text equals the shard line's would be
        no overlap. Overlap is counted on the episodes only (here: none)."""
        sc = self.fresh()
        self.assertEqual(json.loads((self.passes(sc) / "report" / "gate_report.json").read_text())["contamination"]
                         ["exact_overlap"], 0)

    def test_base_rollouts_after_the_base_cut_ignored(self):
        """Base-run episodes after the base cut (orphaned there) are not in the candidate's weights: not counted."""
        sc = self.fresh()
        row = sc.item_rows[0]
        def more(lines):
            extra = json.loads(json.dumps(lines[-1]))
            extra["iteration"] = 3
            extra["episodes"][0].update(init=dict(row["init"]), payload=row["payload"])
            return lines + [extra]
        sc.edit_rollouts(sc.base_run, more)
        rep = json.loads((self.passes(sc) / "report" / "gate_report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["contamination"]["exact_overlap"], 0)

    def test_base_rollouts_missing_refused(self):
        sc = self.fresh()
        (sc.base_run / "rollouts.jsonl").unlink()
        self.refused("EXT-CONTAMINATION", "--stage-out", sc=sc, text="training episodes")


# ---------------------------------------------------------------------------
# EXT-108: seed, pins, profile, last cut
# ---------------------------------------------------------------------------

class TestRunChecks(Sb3Case):

    def test_seed(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(seed=1000))
        self.refused("EXT-LAUNCH", "--launch-record.training_seed", sc=sc)

    def test_framework_version(self):
        sc = self.fresh()
        p = sc.stage_out / "env-freeze.txt"
        p.write_text(p.read_text().replace("stable-baselines3==2.9.2", "stable-baselines3==2.8.0"))
        self.refused("EXT-FREEZE", "--stage-out", sc=sc)

    def test_commit_line(self):
        sc = self.fresh()
        p = sc.stage_out / "env-freeze.txt"
        p.write_text("".join(l + "\n" for l in p.read_text().splitlines() if not l.startswith("#")))
        self.refused("EXT-FREEZE", "--stage-out", sc=sc, text="commit")

    def test_profile(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(profile="smoke"))
        self.refused("EXT-PROFILE", "--launch-record.profile", sc=sc)

    def test_last_cut(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(last_iteration=8))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="last_iteration")


# ---------------------------------------------------------------------------
# EXT-109: arms name their cut's normalizer and the registered environment
# ---------------------------------------------------------------------------

class TestArmEnvironment(Sb3Case):

    def test_normalizer_of_another_cut(self):
        sc = self.fresh()
        arm = build_arm(sc.root / "eval" / "base-bad", sc.item_rows, baseline_score, cut=sc.base,
                        item_set=sc.item_set_hash,
                        meta={"normalizer_sha256": file_sha256(sc.final_cut / "vecnormalize.pkl")})
        self.refused("EXT-ARM", "--eval-baseline", sc=sc, baseline=arm, text="normalizer_sha256")

    def test_env_version_not_registered(self):
        sc = self.fresh()
        arm = build_arm(sc.root / "eval" / "cand-bad", sc.item_rows, baseline_score, cut=sc.final_cut,
                        item_set=sc.item_set_hash, meta={"env_version_sha256": "sha256:" + "3" * 64})
        self.refused("EXT-ARM", "--eval-candidate", sc=sc, candidate=arm, text="env_version_sha256")

    @unittest.skipUnless(HAS_TRL, NO_TRL)
    def test_trl_arms_still_null(self):
        """Area 1 is unchanged: its evaluator row declares neither, so null stays required."""
        row = registry().EVALUATOR_ROWS[TRL_EVALUATOR_ID]
        self.assertFalse(row["normalizer"] or row["env_version"])


# ---------------------------------------------------------------------------
# EXT-110: ext-arm-fill
# ---------------------------------------------------------------------------

class TestArmFill(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-sb3-fill-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = Sb3Scenario(cls.tmp / "sc", n_items=12)
        cls.n = 0

    def arm(self, **meta) -> Path:
        type(self).n += 1
        return build_arm(self.tmp / f"arm-{self.n}", self.sc.item_rows, baseline_score, cut=self.sc.final_cut,
                         item_set=None, model=None, meta={"sampler_hash": None, **meta})

    def fill(self, arm: Path, cut: Path | None = None, adapter_name: str = ADAPTER):
        return run_cli(["ext-arm-fill", "--adapter", adapter_name, "--checkpoint", str(cut or self.sc.final_cut),
                        "--items", str(self.sc.items_dir), str(arm)], cwd=self.tmp)

    def test_fills_and_is_idempotent(self):
        arm = self.arm()
        before_items = (arm / "items.jsonl").read_bytes()
        proc = self.fill(arm)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        meta = json.loads((arm / "eval_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["checkpoint"]["model_hash"], self.sc.cand_model)
        self.assertEqual(meta["item_set_hash"], self.sc.item_set_hash)
        self.assertEqual(meta["sampler_hash"], sha_of(SAMPLER))
        self.assertEqual(meta["created_at"], "2026-11-12T18:20:00Z")
        self.assertIn("filled checkpoint.model_hash", proc.stdout)
        self.assertEqual((arm / "items.jsonl").read_bytes(), before_items)
        once = (arm / "eval_meta.json").read_bytes()
        proc = self.fill(arm)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("verified checkpoint.model_hash", proc.stdout)
        self.assertEqual((arm / "eval_meta.json").read_bytes(), once)

    def refused(self, arm: Path, field: str, **kw):
        before = (arm / "eval_meta.json").read_bytes()
        proc = self.fill(arm, **kw)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        self.assertTrue([e for e in errs if e["field"] == field], f"{field}: {errs}")
        self.assertEqual((arm / "eval_meta.json").read_bytes(), before, "a refusal writes nothing")

    def test_mismatch_refused(self):
        self.refused(self.arm(checkpoint={"model_hash": self.sc.base_model, "base_model_hash": None}),
                     "ARM.checkpoint.model_hash")
        self.refused(self.arm(item_set_hash="sha256:" + "4" * 64), "ARM.item_set_hash")
        self.refused(self.arm(sampler_hash="sha256:" + "5" * 64), "ARM.sampler_hash")

    def test_normalizer_and_env_refused(self):
        self.refused(self.arm(normalizer_sha256=file_sha256(self.sc.base / "vecnormalize.pkl")),
                     "ARM.normalizer_sha256")
        self.refused(self.arm(env_version_sha256="sha256:" + "6" * 64), "ARM.env_version_sha256")

    def test_other_cut_refused(self):
        """The arm evaluated the final cut; naming the base cut gives another normalizer: refused."""
        self.refused(self.arm(), "ARM.normalizer_sha256", cut=self.sc.base)

    def test_incomplete_cut_refused(self):
        bad = self.tmp / "bad-cut"
        shutil.copytree(self.sc.final_cut, bad)
        (bad / "cut.json").unlink()
        self.refused(self.arm(), "--checkpoint", cut=bad)

    @unittest.skipUnless(HAS_TRL, NO_TRL)
    def test_other_adapter_refused(self):
        """An arm of this recipe's evaluator filled through another installed adapter (the TRL recipe's) is refused."""
        self.refused(self.arm(), "ARM.evaluator_id", adapter_name=TRL_ADAPTER)


# ---------------------------------------------------------------------------
# EXT-104, EXT-105: ext-launch for a CPU-only area whose inputs include the base run's output
# ---------------------------------------------------------------------------

class LaunchSb3(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-sb3-launch-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.sc = Sb3Scenario(self.tmp / "sc", n_items=12)
        self.out = self.sc.output_root / "area2-drones" / "cont-2"
        self.out.mkdir()
        (self.tmp / "uv.lock").write_text("# lock stand-in\n", encoding="utf-8")

    def request(self, **overrides) -> dict:
        rec = self.sc.launch_record()
        req = {"area": 4, "label": "mm-a2-drones-full", "host_work": rec["host_work"],
               "runtime": {"kind": "venv_in_container", "image": IMAGE_ID, "venv_lock": str(self.tmp / "uv.lock")},
               "framework": rec["framework"], "training_seed": 2000, "gpus": False, "network": "none",
               "network_reason": None, "user": run_user(),
               # README "Mounts and command shape"; EXT-114: HOME and XDG_CACHE_HOME are paths under /tmp
               "env": {**{n: "1" for n in ENV_NAMES}, "HOME": "/tmp/home", "XDG_CACHE_HOME": "/tmp/cache"},
               "mounts": [m if m["mode"] == "ro" else {**m, "host": str(self.out)} for m in rec["mounts"]],
               "tmpfs": ["/tmp"], "output_dir": str(self.out), "gpu_est_h": 0, "argv": rec["argv"]}
        req.update(overrides)
        return req

    def launch(self, req: dict, *, profile: str = "smoke", smoke: Path | None = None, roots: list | None = None,
               extra: list[str] = ()):
        path = self.tmp / f"request-{len(list(self.tmp.glob('request-*')))}.yaml"
        path.write_text(yaml.safe_dump(req, sort_keys=False), encoding="utf-8")
        out = path.with_suffix(".json")
        args = ["ext-launch", "--adapter", ADAPTER, "--profile", profile, "--request", str(path), "--gpu-log",
                str(self.sc.host_work / "gpu-log.md"), "--out-record", str(out)]
        if smoke is not None:
            args += ["--smoke-record", str(smoke)]
        args += list(extra)
        env = {**self.sc.env()}
        if roots is not None:
            import os
            env[ENV_INPUT_ROOTS] = os.pathsep.join(map(str, roots))
        return run_cli(args, cwd=self.tmp, env=env), out

    def refused(self, proc, out, rule, field):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        self.assertTrue([e for e in errs if e["rule"] == rule and e["field"] == field], f"{rule} {field}: {errs}")
        self.assertFalse(out.exists())


class TestCpuLaunch(LaunchSb3):

    def test_cpu_request_accepted(self):
        """EXT-105: gpus false with gpu_est_h 0; the command has no GPU runtime and estimates 0 GPU-hours."""
        proc, out = self.launch(self.request())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertNotIn("--gpus", proc.stdout)
        self.assertNotIn("--runtime=nvidia", proc.stdout)
        self.assertIn("GATEDTRAIN_GPU_STEP_EST_H=0 ", proc.stdout)
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((rec["gpus"], rec["gpu_est_h"]), (False, 0))

    def test_cpu_request_negative_estimate_refused(self):
        proc, out = self.launch(self.request(gpu_est_h=-0.5))
        self.refused(proc, out, "EXT-LAUNCH", "--request.gpu_est_h")

    def test_gpu_request_with_zero_still_refused(self):
        proc, out = self.launch(self.request(gpus=True, gpu_est_h=0))
        self.refused(proc, out, "EXT-LAUNCH", "--request.gpu_est_h")

    def test_full_run_takes_a_cpu_smoke(self):
        """EXT-105: a CPU full run takes the smoke's projection, 0 GPU-hours; the smoke is judged on the adapter's
        first, deciding evaluator (hover_success, bounded, not binary)."""
        good = write_canonical(self.tmp / "smoke.json", smoke_record(base_model=self.sc.base_model,
                                                                     item_set=self.sc.item_set_hash))
        proc, out = self.launch(self.request(), profile="full", smoke=good)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("GATEDTRAIN_GPU_STEP_EST_H=0 ", proc.stdout)
        degenerate = write_canonical(self.tmp / "smoke-flat.json", smoke_record(
            base_model=self.sc.base_model, item_set=self.sc.item_set_hash, baseline_scores=[0.0] * 12 + [0.5] * 8))
        proc, out = self.launch(self.request(), profile="full", smoke=degenerate)
        self.refused(proc, out, "EXT-SMOKE", "--smoke-record")             # degenerate_mode on hover_success
        self.assertIn("degenerate_mode", proc.stderr)
        negative = write_canonical(self.tmp / "smoke-neg.json", smoke_record(
            base_model=self.sc.base_model, item_set=self.sc.item_set_hash, projected_gpu_h=-1.0))
        proc, out = self.launch(self.request(), profile="full", smoke=negative)
        self.refused(proc, out, "EXT-SMOKE", "--smoke-record")
        gpu = write_canonical(self.tmp / "smoke-gpu0.json", smoke_record(base_model=self.sc.base_model,
                                                                         item_set=self.sc.item_set_hash))
        proc, out = self.launch(self.request(gpus=True, gpu_est_h=0.5), profile="full", smoke=gpu)
        self.refused(proc, out, "EXT-SMOKE", "--smoke-record")             # a GPU run still needs a projection > 0

    def test_spec_optional_and_checked(self):
        """EXT-101, EXT-103: --spec is optional for sb3_ppo (the base phase has no spec yet); given, the base cut and
        the env-params directory must be read-only mounts."""
        proc, out = self.launch(self.request(), extra=["--spec", str(self.sc.paths["spec"])])
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        req = self.request()
        req["mounts"] = [m for m in req["mounts"] if m["container"] != BASE_CONTAINER]
        req["argv"] = req["argv"][:-2]
        proc, out = self.launch(req, extra=["--spec", str(self.sc.paths["spec"])])
        self.refused(proc, out, "EXT-BASE", "artifacts.base.uri")
        proc, out = self.launch(req)                                      # a base-phase shape without --spec
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_cpu_record_negative_estimate_refused(self):
        """The record-side rule (ext-collect, --recheck)."""
        from gatedtrain.ext.launch import host_config, record_problems
        probs = record_problems(self.sc.launch_record(gpu_est_h=-1), adapter=ADAPTER, stage_out=self.sc.stage_out,
                                framework="stable-baselines3", config=host_config(self.sc.env()))
        self.assertIn("--launch-record.gpu_est_h", [p.field for p in probs])
        ok = record_problems(self.sc.launch_record(), adapter=ADAPTER, stage_out=self.sc.stage_out,
                             framework="stable-baselines3", config=host_config(self.sc.env()))
        self.assertNotIn("--launch-record.gpu_est_h", [p.field for p in ok])


class TestInputRootUnderOutputRoot(LaunchSb3):

    def roots(self) -> list:
        """The registry's shape: the runtime, the interpreter (here inside it) and the base runs' output directory
        under the output root."""
        return [self.sc.runtime, self.sc.output_root / "area2-drones"]

    def test_base_cut_under_the_output_root_accepted(self):
        """EXT-104: the base cut lies in another run's output under the output root and is a read-only input."""
        proc, out = self.launch(self.request(), roots=self.roots())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_own_output_as_input_refused(self):
        """EXT-104: a read-only mount of the run's own output directory, or of a directory inside it, is refused."""
        inner = self.out / "sub"
        inner.mkdir()
        for i, host in enumerate((self.out, inner)):
            with self.subTest(host=host):
                req = self.request()
                req["mounts"].append({"host": str(host), "container": f"/inputs/own{i}", "mode": "ro"})
                proc, out = self.launch(req, roots=self.roots())
                self.refused(proc, out, "EXT-LAUNCH", f"--request.mounts[{len(req['mounts']) - 1}].host")
                self.assertIn("EXT-104", proc.stderr)

    def test_parent_of_output_still_refused(self):
        req = self.request()
        req["mounts"].append({"host": str(self.sc.output_root / "area2-drones"), "container": "/inputs/all",
                              "mode": "ro"})
        proc, out = self.launch(req, roots=self.roots())
        self.refused(proc, out, "EXT-LAUNCH", f"--request.mounts[{len(req['mounts']) - 1}].host")

    def test_registry_roots(self):
        """The row names no host path: the runtime directory (this repository's runtime/ or the override), the
        interpreter tree its venv was made from (none before `uv sync`), and the recipe's directory under the host's
        output root, filled in by ext-launch."""
        from gatedtrain.ext import registry as reg
        from gatedtrain_recipe_sb3_ppo_drone_hover import rows
        runtime = rows.RUNTIME
        roots = reg.input_roots(ADAPTER)
        self.assertEqual(roots[-2:], (runtime, "{output_root}/sb3-ppo-drone-hover"))
        self.assertEqual(reg.env_names(ADAPTER), tuple(ENV_NAMES))
        self.assertEqual(reg.scan_skip(ADAPTER)[0], f"{runtime}/.venv")
        self.assertEqual(reg.interpreter(ADAPTER), f"{runtime}/.venv/bin/python")
        tree = rows.python_tree(runtime)
        self.assertEqual(reg.interpreter_trees(ADAPTER), (tree,) if tree else ())
        self.assertEqual(roots[:-2], (str(Path(tree).parent),) if tree else ())

    def test_python_tree_from_pyvenv_cfg(self):
        from gatedtrain_recipe_sb3_ppo_drone_hover import rows
        root = Path(tempfile.mkdtemp(prefix="sb3-rows-"))
        self.addCleanup(shutil.rmtree, root, True)
        self.assertIsNone(rows.python_tree(str(root)))
        (root / ".venv").mkdir()
        (root / ".venv" / "pyvenv.cfg").write_text("home = /opt/py/cpython-3.12.16-linux-x86_64-gnu/bin\n"
                                                    "version_info = 3.12.16\n", encoding="utf-8")
        self.assertEqual(rows.python_tree(str(root)), "/opt/py/cpython-3.12.16-linux-x86_64-gnu")


if __name__ == "__main__":
    unittest.main()
