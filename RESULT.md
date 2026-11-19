Prototype result of a single registered run; not a benchmark claim; not affiliated with or endorsed by the owners of the models, datasets, or frameworks used.

# Result (2026-11-12): gate PASS

stable-baselines3 2.9.2 PPO on gym-pybullet-drones `HoverAviary` (commit `15c426c8`, launcher-side random initial
states), CPU only (0 GPU-hours; base 23 s, continuation 85 s wall). Base run cut 8 (pinned in the spec),
continuation 50 iterations to cut 58.

| | |
|---|---|
| Deciding metric | `hover_success` (T3: altitude within 0.06 m, the CF2X collision radius) |
| Baseline (base cut 8) | 0.322 |
| Candidate (cut 58) | 0.992 |
| Delta, 90% cluster-bootstrap interval | +0.669 [0.631, 0.705] |
| Items, samples per item | 300, 1 |
| Achieved power | 0.849 |
| Exact overlap with training | 0 |
| Consumption precision | `iteration` (observed) |

The impact query on the environment shard returns the continuation's cuts 18 to 58 (first step 9); the base phase's
cuts are not returned (the base run is not collected).

Re-collected on 2026-11-16 with this repository's adapter (installed as a plugin of ptc; ptc, ptc-skel and ptc-hub are earlier
names of gatedtrain, see "Rename to gatedtrain 0.3.0") on the original run's
outputs: every field of the gate report is unchanged; the ledger and lineage differ only in their run-id-derived
columns.

## Published copies

The files under results/2026-11-12/ are redacted copies (host paths replaced with `<data-root>/...`, internal
process notes reworded) of `gate/EVAL-PLAN.md`, `gate/registration.yaml` and `gate/spec.yaml`; the gate-time
originals are not published.

`trainer.yaml` is unchanged. `spec.yaml` still pins the original `registration.yaml` hash (the gate ran with it).

`runtime/eval_hover.py` (the scorer, `code_sha256` `sha256:8bc66417...` in the registry rows) was unchanged by the
publication cleanup; the renames to ptc-hub 0.2.0 and gatedtrain 0.3.0 (below) changed it by the name map only.
`runtime/train_ppo.py` differs from the script that produced the result (not published) only by the publication
cleanup of two comments and the rename to gatedtrain 0.3.0 (below); the result was re-collected unchanged (above) before the cleanup.

## Rename to ptc-hub 0.2.0 (2026-11-17)

ptc-hub 0.2.0 renamed the persisted identifiers (schema ids, ledger writer id, analysis code version). The files under
results/2026-11-12/ keep the names ptc-skel 0.1.0 wrote (`registration.yaml`, `spec.yaml`, `EVAL-PLAN.md` and
`trainer.yaml` are the gate's historical files); they are read by ptc `78eb913` (this and the other framework commits cited here, `e20736e` among them,
are unpublished pre-release commits of the framework, earlier named ptc-skel and ptc-hub; the published framework is
`gatedtrain` `v0.3.0`), with which the historical verdict
remains reproducible from the original `registration.yaml`/`spec.yaml`. The gate definition is carried forward by
results/2026-11-12/gate/registration-ptc-hub.yaml (amends
sha256:10b543ddf3c331f10f361aa0de727fa94d87d1a3aa89198b7de36808a524f23d, the published `registration.yaml`;
post_hoc: false) and spec-ptc-hub.yaml. Re-pinned for the rename: `runtime/eval_hover.py`
`8bc66417df218995c8a47eea8489a2e66369aede4a10589bc80a8111136adb2a` ->
`ff0cdc0a902aaa6c1191c655959b55381efb958fc32cf566bce31dc4f7120c4b` (only the new pin is accepted; the arms of the
2026-11-12 run record the old one).

- Scorer (design revision 17, R17-3): the new `eval_hover.py` is exactly ptc's `renamed.apply_name_map` of the
  gate-time file, kept byte for byte as `results/2026-11-12/gate/scorer-before/eval_hover.py` (SHA-256 `8bc66417...`,
  from git). The evaluator `drones.sb3_items@0.1` therefore has a new identity: `manifest_hash`
  `sha256:0069196c...` -> `sha256:1787b6234456d1156f80106a904746094663cde10058da038322aabee54385ff`, proved by the
  amendment's `rename_proof` (ptc checks it at validation). No other registered value changed (sampler,
  `eval_distribution`, other evaluator fields); `runtime/train_ppo.py` and the env snapshot are untouched.
- Files: `registration-ptc-hub.yaml` (SHA-256 `c8692c22d122f2f48e44af37a061d76ffff07a69656ebb7c3577d442541f8d42`)
  differs from `registration.yaml` only in `analysis_code_version: ptc-hub 0.2.0`, `amends`, `created_at`
  (2026-11-17T17:49:12Z), the evaluator's `manifest_hash` and its comment, `rename_proof`, and the amendment comment;
  `spec-ptc-hub.yaml` (SHA-256 `20241805a7c419d9176157ea0b071b725e10416d9e4824c42d05636e62be1554`) differs from
  `spec.yaml` only in the registration uri and the name map in comments. `ptc-hub validate` accepts it (with the
  `<data-root>` placeholders filled in); the original spec under ptc-hub 0.2.0 is refused with
  `error[REG-FIELD] registration.analysis_code_version` naming the amendment.
- Re-judged on a scratch replay with ptc `e20736e`. Copies of the run's launch record (`full-1.launch.json`), smoke
  record and both arms had only their schema ids replaced (`ptc_hub.launch_record`, `ptc_hub.smoke_record`,
  `ptc_hub.item_eval`) and, in each arm's `eval_meta.json`, `scorer.code_sha256` `8bc66417...` -> `ff0cdc0a...` (the
  pair the rename proof proves). The launch record's label is a checksum of the record, and gpu-step.sh logs that
  label in the host's `gpu-guard.log`, so the replay record was moved to a scratch host-work directory holding a copy
  of the host's `gpu-step.sh` (its `WORK=` line changed) and the run's two guard-log lines under the re-hashed
  label; `host_work`, `gpu_step_sha256`, `host_config_sha256` (original input roots) and `label` were recomputed with
  ptc's own functions. The stage output is the original. `ext-collect` on `spec-ptc-hub.yaml` reported
  `registration_amendment: version_only`, `registration_amends: sha256:10b543dd...`, `post_hoc: false` and, against
  the canonical 2026-11-12 report: status PASS, no status reasons, delta_hat 0.66945, the interval, p value
  1.594e-04, achieved power 0.8493, n 300, scores 0.32219 / 0.99164, item-set and sampler hashes, model hashes,
  contamination, consumption precision, GPU and wall seconds all equal. Differences, all expected: `schema`,
  `analysis_code_version`, `registration_hash`, `registration_amends`, `registration_amendment` (new),
  `evaluator_hash` and `evaluator_hash_registered` (the R17-3 identity above), `lockfile_hash`, `run_id`,
  `created_at`. Ledger (50 rows): equal except `writer_id` (`ptc-hub.ledger@0.2.0`) and the run-id-derived `run_id`
  and `sample_id`. Lineage: equal except run ids, the registration hash and its schema id.

## Rename to gatedtrain 0.3.0 (2026-11-18)

The framework was renamed again, from ptc-hub to gatedtrain, before its first publication (distribution, import
package and CLI `gatedtrain`, entry-point group `gatedtrain.adapters`, schema ids `gatedtrain.<x>`, ledger writer id
`gatedtrain.ledger@0.3.0`, analysis code version `gatedtrain 0.3.0`, environment variables `GATEDTRAIN_<X>`); this
repository became `gatedtrain-recipe-sb3-ppo-drone-hover` (import package `gatedtrain_recipe_sb3_ppo_drone_hover`,
runtime directory variable `GATEDTRAIN_RECIPE_SB3_PPO_DRONE_HOVER_RUNTIME`). The sections above keep the names in use
when they were written (ptc, ptc-skel, ptc-hub). The files under results/2026-11-12/ keep their bytes, including
`registration-ptc-hub.yaml`, `spec-ptc-hub.yaml` and `scorer-before/`, and so do the original host's directory names
the specs cite (`<data-root>/ptc-mm/...`); the historical verdict remains reproducible with the commit that wrote each
(ptc `78eb913` for `registration.yaml`/`spec.yaml`, ptc-hub `e20736e` for the `-ptc-hub` pair). The registered item
set (`runtime/items/`, whose metadata names its schema `ptc_mm.drones_item_set`) and the environment snapshot
(`runtime/env-snapshot/`) are unchanged: their hashes are registered.

- Scorer (05 section 3.7, R17-3): the new `runtime/eval_hover.py` is exactly gatedtrain's `renamed.apply_name_map` of
  the ptc-hub-era file, kept byte for byte as `results/2026-11-12/gate/scorer-ptc-hub/eval_hover.py` (SHA-256
  `ff0cdc0a...`, from git). It still writes the item-set schema id `ptc_mm.drones_item_set`, which the map does not
  cover; any other edit would make the amendment not version-only. The evaluator `drones.sb3_items@0.1` therefore
  has a new identity: `manifest_hash` `sha256:1787b623...` ->
  `sha256:e635687e3005bd24190e53a326e8c2792b4ed50de0d58c87409b143231dc7f49`, proved by the amendment's
  `rename_proof` (gatedtrain checks it at validation; a scratch copy with one byte appended to the kept scorer is
  refused with `error[REG-FIELD] registration.post_hoc`). The `HoverItemEnv` block that `env_class_sha256` hashes is
  byte-identical in `train_ppo.py` and `eval_hover.py` (`sha256:fa3efa3a...`, as `env_version.json` registers).
- Files: `registration-gatedtrain.yaml` (SHA-256 `e38be3d3da9db586a54bc51cdca6014da1dfb02de21ee79dba2efa8a11b4a5a4`)
  differs from `registration-ptc-hub.yaml` only in `analysis_code_version: gatedtrain 0.3.0`, `amends`
  (`sha256:c8692c22...`, the `-ptc-hub` registration), `created_at` (2026-11-18T09:52:36Z), the evaluator's
  `manifest_hash` and its comment, `rename_proof` (`code_sha256_before` `sha256:ff0cdc0a...`, files
  `scorer-ptc-hub/eval_hover.py`) and the amendment comment; `post_hoc: false`. `spec-gatedtrain.yaml` (SHA-256
  `7eb405c0171d7740f77c95c6939642b3b27ec742a9fa77e3900cfd7acc642540`) differs from `spec-ptc-hub.yaml` only in the
  registration uri and the name map in comments. `gatedtrain validate` accepts it (with the `<data-root>`
  placeholders filled in); `spec-ptc-hub.yaml` is refused with `error[REG-FIELD] registration.analysis_code_version`
  naming the amendment.
- Re-judged on a scratch replay with gatedtrain 0.3.0, starting from the scratch replay of the ptc-hub rename
  (above). Copies of its launch record, smoke record and both arms had only their schema ids replaced
  (`gatedtrain.launch_record`, `gatedtrain.smoke_record`, `gatedtrain.item_eval`) and, in each arm's
  `eval_meta.json`, `scorer.code_sha256` `ff0cdc0a...` -> `e067719e...` (the pair the rename proof proves). The record
  moved to a new scratch host-work directory as before (`host_work`, `gpu_step_sha256`, `host_config_sha256` with the
  original input roots, and `label` recomputed with gatedtrain's own functions; the run's two guard-log lines under
  the re-hashed label). The stage output is the original, read in place. `gatedtrain ext-collect` on
  `spec-gatedtrain.yaml` reported `registration_amendment: version_only`, `registration_amends: sha256:c8692c22...`,
  `post_hoc: false` and, against the canonical 2026-11-12 report: status PASS, no status reasons, delta_hat 0.66945,
  interval [0.63131, 0.70483], p value 1.594e-04, achieved power 0.8493, n 300, scores 0.32219 / 0.99164, item-set
  and sampler hashes, model hashes, contamination (exact overlap 0), consumption precision `iteration`, GPU seconds
  (0) and wall seconds (85) all equal. Differences, all expected: `schema`, `analysis_code_version`,
  `registration_hash`, `registration_amends`, `registration_amendment`, `evaluator_hash` and
  `evaluator_hash_registered` (the R17-3 identity above), `lockfile_hash`, `run_id`, `created_at`. Ledger (50 rows):
  equal except `writer_id` (`gatedtrain.ledger@0.3.0`) and the run-id-derived `run_id` and `sample_id`. Lineage:
  equal except run ids, the registration hash and its schema id.
- Re-pinned for the rename: `runtime/eval_hover.py` `ff0cdc0a902aaa6c1191c655959b55381efb958fc32cf566bce31dc4f7120c4b`
  -> `e067719e6aa6418dec61afa86f82502d5d89ebdb44e48de0d748fae6225960a7` (registry rows; only the new pin is accepted,
  the arms of the 2026-11-12 run record the old one). Not registered, recorded by ext-launch for a new run:
  `runtime/train_ppo.py` `eed8af4ead8722bbf8882504fc902502f609e0e482c0a05dee8f4d5cb5467fb3` ->
  `4e5898988b20a5a714f85ec292670fd4a282f59bf3e2a84b51746d36dcdbe29c` (comments, the callback class name, the
  `cut.json` and `run_args.json` schema ids `gatedtrain_mm.sb3_ppo_cut` and `gatedtrain_mm.sb3_ppo_run_args`);
  `runtime/uv.lock` `82cef6175f94620c66bb7a37f8d8db9eaf8d59ab7a3e9facbbd52adc7c73e272` ->
  `10316d5a0f96aed4d2900422a4755a6a0cf25e8031ad40d036d7947faab3db15` (regenerated with `uv lock` after the runtime
  project was renamed to `gatedtrain-recipe-sb3-ppo-drone-hover-runtime`; the 36 locked packages and their versions,
  sources and hashes are unchanged, only the root project's entry moved).
