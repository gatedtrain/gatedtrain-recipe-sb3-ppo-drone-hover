"""gatedtrain's generic launch, fill and collect rules exercised through the `sb3_ppo` adapter (moved here from
gatedtrain's review round 6 and 9 and SD1.5 tests when the recipe left gatedtrain): container paths (EXT-112), the
interpreter pin (EXT-113), path-valued environment names (EXT-114), unreadable files in the generic paths (review 9 M1)
and fill's sampler-hash conversion (EXT-137)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from _sb3 import ADAPTER as SB3_ADAPTER, SAMPLER as SB3_SAMPLER, Sb3Scenario, baseline_score, build_arm
from gatedtrain.testing.ext import sha_of
from gatedtrain.testing.support import EXIT_OK, EXIT_REFUSED, describe, parse_errors, run_cli
from test_adapter import LaunchSb3

# EXT-112: container paths a reviewer named; a mount there replaces the image's loader, entrypoint or libraries.
REVIEWER_CONTAINERS = ("/etc/ld.so.preload", "/opt/nvidia/nvidia_entrypoint.sh", "/usr/lib/python3")
PATH_ENV = ("HOME", "XDG_CACHE_HOME")                  # runtime/README.md "Mounts and command shape"


def compact_hash(obj) -> str:
    """The runtime's own hash encoding (`canonical` in its scripts): compact sorted JSON, not RFC 8785."""
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()


class TestRegistryRows(unittest.TestCase):

    def test_container_paths_and_env_paths(self):
        from gatedtrain.ext import registry
        self.assertEqual(dict(registry.container_paths(SB3_ADAPTER)),
                         {"ro_identity": True, "ro_under": ("/inputs",), "rw": ("/out",), "tmpfs": ("/tmp",)})
        self.assertEqual(registry.env_paths(SB3_ADAPTER), PATH_ENV)
        self.assertLessEqual(set(registry.env_paths(SB3_ADAPTER)), set(registry.env_names(SB3_ADAPTER)))

    def test_host_interpreter_lies_in_its_tree(self):
        from gatedtrain.ext import registry
        path = registry.interpreter(SB3_ADAPTER)
        if not os.path.exists(path):
            self.skipTest(f"{path} absent (run `uv sync` in runtime/)")
        real = os.path.realpath(path)
        self.assertTrue(any(real.startswith(t.rstrip("/") + "/") for t in registry.interpreter_trees(SB3_ADAPTER)))


class TestContainerPathsSb3(LaunchSb3):

    def test_reviewer_mounts_refused(self):
        for container in REVIEWER_CONTAINERS:
            with self.subTest(container=container):
                req = self.request()
                req["mounts"].append({"host": str(self.sc.runtime / "train_ppo.py"), "container": container,
                                      "mode": "ro"})
                proc, out = self.launch(req)
                self.refused(proc, out, "EXT-LAUNCH", f"--request.mounts[{len(req['mounts']) - 1}].container")

    def test_record_mode(self):
        from gatedtrain.ext.launch import host_config, record_problems
        for container in REVIEWER_CONTAINERS:
            with self.subTest(container=container):
                rec = self.sc.launch_record()
                rec = self.sc.launch_record(mounts=rec["mounts"] + [{"host": str(self.sc.runtime / "train_ppo.py"),
                                                                     "container": container, "mode": "ro"}])
                got = [p.field for p in record_problems(rec, adapter=SB3_ADAPTER, stage_out=self.sc.stage_out,
                                                        framework="stable-baselines3",
                                                        config=host_config(self.sc.env()))]
                self.assertIn(f"--launch-record.mounts[{len(rec['mounts']) - 1}].container", got)
        ok = record_problems(self.sc.launch_record(), adapter=SB3_ADAPTER, stage_out=self.sc.stage_out,
                             framework="stable-baselines3", config=host_config(self.sc.env()))
        self.assertEqual([p.field for p in ok], [])


class TestInterpreterPinSb3(LaunchSb3):

    def test_other_interpreter_refused(self):
        other = self.sc.runtime / "other" / "bin" / "python"
        other.parent.mkdir(parents=True)
        other.write_text("# stub\n", encoding="utf-8")
        req = self.request()
        req["argv"] = ["/inputs/a2/other/bin/python", *req["argv"][1:]]
        proc, out = self.launch(req)
        self.refused(proc, out, "EXT-LAUNCH", "--request.argv")


class TestEnvPathsSb3(LaunchSb3):

    def test_refused(self):
        for name in PATH_ENV:
            with self.subTest(name=name):
                req = self.request()
                req["env"][name] = "/inputs/a2"
                proc, out = self.launch(req)
                self.refused(proc, out, "EXT-LAUNCH", f"--request.env.{name}")


class TestUnreadable(unittest.TestCase):
    """Review 9 M1 (EXT-154): an unreadable file in the generic paths is a refusal, never an exception."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sb3-review9-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def unreadable(self, p: Path) -> None:
        if os.geteuid() == 0:
            self.skipTest("running as root: chmod 0 does not make a file unreadable")
        p.chmod(0)
        self.addCleanup(p.chmod, 0o644)

    def refused(self, proc, rule: str, field_prefix: str | None = None, text: str | None = None):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))
                and (text is None or text in e["message"])]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix} {text!r}; got {errs}\n{describe(proc)}")

    def test_sb3_freeze_and_cut(self):
        sc = Sb3Scenario(self.tmp / "s1", n_items=12)
        self.unreadable(sc.stage_out / "env-freeze.txt")
        self.refused(sc.collect(self.tmp / "runs1"), "EXT-FREEZE", "--stage-out")
        sc = Sb3Scenario(self.tmp / "s2", n_items=12)
        self.unreadable(sc.final_cut / "policy_meta.json")
        self.refused(sc.collect(self.tmp / "runs2"), "EXT-STAGE", "--stage-out", "policy_meta.json cannot be read")


class TestConvertedSampler(unittest.TestCase):
    """EXT-137 (review L5): fill converts the arm's own sampler-hash encoding for an sb3_ppo arm too."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sb3-sampler-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_sb3_arm_converted(self):
        sc = Sb3Scenario(self.tmp / "sb3", n_items=12)
        floaty = {**SB3_SAMPLER, "max_steps": 242.0}           # compact "242.0" differs from RFC 8785 "242"
        self.assertNotEqual(compact_hash(floaty), sha_of(floaty))
        arm = build_arm(self.tmp / "sb3-arm", sc.item_rows, baseline_score, cut=sc.final_cut, item_set=None, model=None,
                        meta={"sampler": floaty, "sampler_hash": compact_hash(floaty)})
        proc = run_cli(["ext-arm-fill", "--adapter", "sb3_ppo", "--checkpoint", str(sc.final_cut), "--items",
                        str(sc.items_dir), str(arm)], cwd=self.tmp)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("converted sampler_hash", proc.stdout)
        self.assertEqual(json.loads((arm / "eval_meta.json").read_text())["sampler_hash"], sha_of(floaty))


if __name__ == "__main__":
    unittest.main()
