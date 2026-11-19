# Area "control" (SB3 PPO hover) gate: evaluation plan (DRAFT)

prototype, unpromotable. Draft written 2026-11-12 (UTC) on the ptc-skel side, before the area's base run. The
operator writes the real files under `<data-root>/ptc-mm/area2-drones/gate/` (this draft never
touches the data directory). Every step is CPU only (`gpus: false`); each still runs through `ext-launch` and
`gpu-step.sh` after approval (MM-31), and `ext-collect` logs it with 0 GPU-seconds (EXT-105).

## 1. Files

| File | Purpose | Hash |
|---|---|---|
| `spec.yaml` | workflow `mm-a2-drones-hover`: stage `rl.sb3_ppo@0.1` (the continue phase), gate `gate.superiority@1.0` on `drones.sb3_items@0.1` | pins env_params, items, trainer config, registration; **base pinned after the base run** |
| `registration.yaml` | v1; hover_success, superiority, margin 0, sigma 0.39 (pilot), alpha 0.05 one-sided, power 0.8, planning effect 0.06 (n_required 262 of 300), K 1, eval seed 0; `created_at` to be added when written | re-pin in spec.yaml after adding `created_at` |
| `trainer.yaml` | `{backend: sb3_ppo, profile: full}` | file SHA-256 in spec.yaml |

Fixed hashes (computed 2026-11-12 with ptc-skel, equal to the area2-drones README):
- item set `area2-drones/items/` manifest `sha256:8531e5ba14941466275659cc4c0381376acd17caf4b5d31b36b8e78948daf7e7` (300 items);
- env-params shard directory `area2-drones/env-snapshot/shards/` manifest
  `sha256:d12f691ef99baa3bfffb7a3bc4ddaeefc429d17975e60057ccd1a4ae4d532bbc` (the spec's training data, C3; the
  ledger's `data_ref.snapshot`); whole snapshot `sha256:f4e0fa91...` (not used by ptc-skel);
- `env-snapshot/env_version.json` `sha256:c0e03e0ebf22a8a726d823cd4d0505bdc41149ba0118aca11f05b45c482d2fb2`
  (registration `eval_distribution.env_version`; every cut's copy must equal it);
- evaluator identity `drones.sb3_items@0.1` `sha256:0069196c5011c3937131b249c09d346eb67f8df848e702fc75080f5eb2d235ff`
  (scorer `code_sha256` = SHA-256 of `eval_hover.py`, `sha256:8bc66417...`; changing `eval_hover.py` changes the
  registry row and this hash); supporting `drones.sb3_return_items@0.1` `sha256:7e1657ef...` (not registered:
  descriptive only).

## 2. Open items before the gated run

1. **Base pin.** After the base run, pin `base` in spec.yaml with the T0-09 manifest hash of
   `out/area2-drones/base-1/checkpoint-8` (`ptc-skel` `canonical.manifest_hash`), and record its sb3 `model_hash`
   (`ext-arm-fill` prints it as `checkpoint.model_hash` for the baseline arm).
2. **Base launch profile.** `ext-launch --profile full` needs a PASS smoke record, whose baseline scores are the
   base cut's, which exists only after the base run. Proposal: launch the base phase with `ext-launch --profile
   smoke` (argv `--phase base --profile full`); its record is never collected (no gate is on the base run), and the
   smoke record gates the continuation's full run (MM-39).
3. **MM-38 margin (README open question 1).** 49.0 % of the base cut's 300 scores are 0; the smoke's 20 items must
   stay under 50 % (scratch e2e on items 0-19: 45 %, PASS).
4. **Registration `created_at`** must precede the continuation full run's first `START` in `gpu-guard.log`.

## 3. Runs (CPU; request shapes as the area2-drones README "Request shapes")

Input roots of `sb3_ppo` (registry): `<data-root>/ptc-mm/python`, `.../ptc-mm/area2-drones`,
`.../ptc-mm/out/area2-drones` (the base run's output is the continuation's read-only input, under the output root;
a read-only mount at or inside the run's own output is refused, EXT-104). Environment: exactly `HF_HUB_OFFLINE`,
`HOME`, `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`, `XDG_CACHE_HOME`. `gpus: false`, `gpu_est_h: 0`.

1. **Base phase:** request label `mm-a2-drones-base`, training_seed 1000, output `out/area2-drones/base-1` (new,
   empty, owned by the launching user), argv `[<venv python>, <train_ppo.py>, --phase, base, --profile, full, --out, /out,
   --env-snapshot, <area2-drones>/env-snapshot]`. `ext-launch --adapter sb3_ppo --profile smoke` (no `--spec`: the
   base has no spec yet; `--spec` is optional for this adapter). Expect cuts 4 and 8, `rollouts.jsonl` iterations 1-8.
2. **Smoke record:** `eval_hover.py --checkpoint <base cut 8> --limit 20` on the registered items, then
   `smoke.json` (`adapter sb3_ppo`, `area 4`, `base_model_hash` = the base cut's sb3 model hash, `item_set_hash`
   `sha256:8531e5ba...`, `peak_gpu_mem_gb 0`, `projected_gpu_h 0`, the 20 hover_success scores, `error_count 0`);
   the verdict is recomputed by ptc-skel (judged on the deciding evaluator, bounded, not binary).
3. **Continuation smoke and full:** label `mm-a2-drones-smoke` / `mm-a2-drones-full`, training_seed 2000, mount
   `<base-1>/checkpoint-8` read-only at `/inputs/base`, argv `... --phase continue --profile smoke|full ... --base
   /inputs/base`. `ext-launch --adapter sb3_ppo --profile full --smoke-record smoke.json --spec <gate>/spec.yaml`,
   then `ext-launch --adapter sb3_ppo --recheck <record> --spec <gate>/spec.yaml` right before the printed command.
   Expect cuts 18, 28, 38, 48, 58; `run_args.json` `first_iteration 9`, `last_iteration 58`, `base_checkpoint
   /inputs/base`, `base_cut` = cut 8's `cut.json`, `seed 2000`.

## 4. Arms (CPU)

`eval_hover.py --checkpoint /inputs/cut --items <area2-drones>/items --env-snapshot <area2-drones>/env-snapshot
--arm-out /out --eval-seed 0 --workers 16 --metric hover_success` once on base cut 8 (baseline) and once on cut 58
(candidate); a second candidate run with `--metric hover_score` is the supporting arm (not collected). Then, on the
host, for each arm:

```
ptc-skel ext-arm-fill --adapter sb3_ppo --checkpoint <the evaluated cut> --items <data-root>/ptc-mm/area2-drones/items <arm>
```

It writes `checkpoint.model_hash`, `item_set_hash` and `sampler_hash` where they are null, verifies them where set,
and refuses (exit 3, nothing written) on any mismatch or when the arm's `normalizer_sha256` or `env_version_sha256`
is not the cut's.

## 5. Collect

```
ptc-skel ext-collect <gate>/spec.yaml --adapter sb3_ppo --stage-out <data-root>/ptc-mm/out/area2-drones/full-1 --launch-record <full-1 launch.json> --smoke-record <smoke.json> --eval-baseline <baseline arm> --eval-candidate <candidate arm>
```

No `PTC_SKEL_EXT_*` override may be set. **Expect:** `metric: hover_success`, `independence_tier: T3`, `n_items 300`,
`samples_per_item 1`, unpaired 0/0, `baseline_score` near 0.322, `gpu_seconds: 0`, `wall_seconds` the wrapper time,
`contamination: exact_overlap=0` (items against every episode payload of base iterations 1-8 and continuation
iterations 9-58), `consumption_precision: iteration` (observed; ledger steps 9-58 only), `baseline_model_hash` = the
base cut's sb3 model hash. `ptc-skel impact <area2-drones>/env-snapshot/shards/hover-init-v1.jsonl --run-dir <run>`
returns cuts 18-58 with first step 9; base cuts 4 and 8 are not returned (the base run is not collected; §7).

## 6. Scratch end-to-end check (2026-11-12, the run host's CPU, outputs in a temporary directory)

Real `train_ppo.py` and `eval_hover.py` in the area venv, run directly (no docker): base phase full (8 iterations,
cuts 4 and 8), continuation of 10 iterations (`--iterations 10 --save-every 5`, cuts 13 and 18), both arms on the
first 30 registered items, `ext-arm-fill`, `ext-collect` under this draft with the item-set hash of the 30 items.
Result: **INCONCLUSIVE** (underpowered: n 30 < n_required 262); Δ̂ −0.300, 90 % interval [−0.420, −0.201]
(cut 18 after only 10 continue iterations settles about 0.16 m off the target height: hover_success 0 on every item,
hover_score 0.992; base cut 8: 0.300); `gpu_seconds 0`, `wall_seconds 19`; `exact_overlap 0` against 677 distinct
training states; precision `iteration` (observed), ledger steps 9-18. Simulated: the container (launch records
written by `ext-launch`, commands not run), `gpu-step.sh` (guard-log START/END lines written around each direct run),
the base cut's container path (its host path; see the e2e request), and no `/tmp` tmpfs in the e2e continuation
record (the scratch paths lie under `/tmp`).

## 7. Lineage of the base phase

The continuation's ledger holds iterations 9-58 only; its training stage `used` base cut 8 (cut hash and model
hash). 05 §11.3 continues the closure across runs through `used` model edges, so base cuts 4 and 8 and the base run's
own consumption (iterations 1-8) appear in impact answers only when the base run has its own run directory with a
ledger; `ext-collect` needs a gate, which the base run does not have. Until this is decided,
an impact query on `hover-init-v1.jsonl` returns the continuation's cuts and not the baseline.
