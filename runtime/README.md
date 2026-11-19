# Area "control" runtime: SB3 PPO, gym-pybullet-drones hover (prototype, unpromotable)

Multimodal prototype, scope row 4 "Control" (second in the run order, hence the original directory name
`area2-drones`), gatedtrain's `MULTIMODAL-SPEC.md` §5-§7, §6 row 4. Runtime kind `venv_in_container` (MM-32): a
pinned CUDA base image plus this directory's host venv, built from the committed `uv.lock` and mounted read-only.
Nothing here starts a container; an operator runs the commands below through `ext-launch` and `gpu-step.sh`
after approval.

## Pins

| Item | Value |
|---|---|
| Base image | the area-1 image, `docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266` (already pulled for area 1; no CUDA is used, any pinned Ubuntu 24.04 image would do) |
| Python | CPython 3.12.16, uv-managed, `<data-root>/python/cpython-3.12.16-linux-x86_64-gnu` (shared with area 1, unchanged) |
| Lock | `uv.lock` (36 packages, `exclude-newer = 2026-10-29`, Linux x86_64), sha256 `10316d5a…3db15` (the published runs' lock, `82cef617…3e272`, differs only in the project name; `../RESULT.md`) |
| Venv | `.venv` (`uv sync --frozen --compile-bytecode`, copy link mode, 1.6 GB) |
| Framework | stable-baselines3 2.9.2 (2026-10-26), gymnasium 1.4.0 (2026-10-05), torch 2.15.0+cpu (2026-10-28, PyTorch CPU index), numpy 2.5.4, pybullet 3.2.7 (built from the locked sdist: PyPI has no cp312 wheel; host gcc 13.3 on Ubuntu 24.04, the image's distribution), gym-pybullet-drones 2.2.0 at commit `15c426c85ac8e7a5db66d8b100f48858133357f6` (2026-10-28) of github.com/learnsyslab/gym-pybullet-drones (formerly utiasDSL; not on PyPI) |
| Citations | installed files under `.venv/lib/python3.12/site-packages/`; the installed `gym_pybullet_drones` is byte-identical to the commit (diff -r) |

Rebuild (host, no GPU): `UV_PROJECT_ENVIRONMENT=$PWD/.venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
uv sync --frozen --compile-bytecode --python <data-root>/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12`.
The venv's `bin/python` links to the absolute interpreter path, so the venv and the Python directory are mounted at
their host paths.

## CPU, not GPU

PPO with a 2 × 64 MLP on 27-dimensional observations is bound by the simulator and by small-batch Python overhead;
CUDA would only add transfers, so the lock carries the CPU build of torch and the policy runs with `device="cpu"`.
The run still goes through `ext-launch` and `gpu-step.sh` (MM-31) with `gpus: false`: the printed command then has no
`--runtime=nvidia` / `--gpus` (launch.py `build_command`, `if record["gpus"]`). Measured on a 32-core host:
2.1-2.7 s per PPO iteration of 8 × 1024 env steps (rollout plus 10 epochs of 32 minibatches), about 330 MB RSS for the
main process and each of the 8 env workers. GPU-hour accounting: CPU steps are logged as 0 GPU-hours (ext-collect side).

## Environment, items, metric

**Environment.** `HoverItemEnv` (in `train_ppo.py`, a byte-identical copy in `eval_hover.py`, hashed as
`env_class_sha256`) subclasses `HoverAviary` with `obs=kin`, `act=one_d_rpm` (examples/learn.py:43, the example's
default), PyBullet at 240 Hz, control at 30 Hz. Stock facts:
- observation `Box(-inf, inf, (1, 27))`: position, roll/pitch/yaw, velocity, angular velocity (12) plus the last
  `ctrl_freq // 2 = 15` actions (BaseRLAviary.py:66, :335-339); action `Box(-1, 1, (1, 1))`, all four motors at
  `HOVER_RPM * (1 + 0.05 a)` (:223-224);
- reward `max(0, 2 - ||(0,0,1) - pos||^4)` per control step (HoverAviary.py:77-79); terminated when the distance is
  below 1e-4 (:93-94); truncated when |x| or |y| > 1.5, z > 2, |roll| or |pitch| > 0.4 (:110-113), or by time,
  `step_counter / 240 > 8` (:114), which with the counter advanced after the check (BaseAviary.py:377-381) gives at
  most **242 control steps** (measured);
- no initial-state randomization: every episode starts at `(0, 0, 0.1125)` with zero attitude
  (BaseAviary.py:193-205), and `reset(seed)` only seeds `np_random` (gymnasium core.py:170-171, called at
  BaseAviary.py:242); the action buffer is never cleared, so an episode's first observation carries the previous
  episode's actions.

`HoverItemEnv` therefore (1) draws the initial state per episode from `np_random`: x, y ~ U[-0.25, 0.25],
z ~ U[0.15, 1.85], vertical velocity vz ~ U[-0.5, 0.5], zero attitude, each rounded to 4 decimals, the rounded
value applied (`env_params_version` **`hover-init-v1`**); (2) or applies an explicit `options={"init": {...}}`
(evaluation); (3) clears the action buffer at every reset; (4) logs each episode's env, per-env episode index, root
seed, initial state and steps. With `one_d_rpm` x and y cannot be corrected; they only shift the reachable reward.

**Environment snapshot (design rev 11 C3, 05 §12.1, 08 §6.1).** `env-snapshot/` is the DatasetSnapshot:
`shards/hover-init-v1.jsonl` (one line, the parameters above; sha256 `3f36545e…36dbf`) and `env_version.json`
(MM-19: package, version, commit, env id, kwargs, assets hash, `env_class_sha256`; sha256 `c0e03e0e…d2fb2`).
`train_ppo.py --write-env-snapshot DIR` wrote it; both scripts refuse a snapshot that does not describe their
runtime. gatedtrain manifest hashes: `env-snapshot/` `sha256:f4e0fa91fba1456cead09b4c7dca2002e65ad189d88fa9abcf180532cdfa8f43`,
`env-snapshot/shards/` `sha256:d12f691ef99baa3bfffb7a3bc4ddaeefc429d17975e60057ccd1a4ae4d532bbc`.

**Item (C5, 05 §3.3 "an episode seed with environment parameters").** One registered initial state. The payload is
the state itself, so scoring needs no RNG. `items/items.jsonl`: **N = 300** items `hover-init-v1-0000` …
`-0299`, fields `item`, `env_params_version`, `init {x, y, z, vz}`, `payload` (canonical text
`hover-init-v1 x=… y=… z=… vz=…`). Generated by `eval_hover.py --make-items --n 300 --item-seed 20261112` from
`numpy.random.default_rng(SeedSequence([0x5EED17E5, 20261112]))`, uniform draws in the snapshot's ranges, rounded,
duplicates skipped (`items/item_set_meta.json`). Item set manifest hash (gatedtrain `canonical.manifest_hash` of
`items/`): `sha256:8531e5ba14941466275659cc4c0381376acd17caf4b5d31b36b8e78948daf7e7`.

**Deciding metric `hover_success`** (evaluator `drones.sb3_items@0.1`, tier T3, higher is better, range **[0, 1]**,
not a function of the reward; chosen 2026-11-12): the fraction of the second half of the control
steps, steps 121 … 242 (122 steps), in which the drone is inside the env's non-truncation bounds (|x|, |y| ≤ 1.5,
z ≤ 2, |roll|, |pitch| ≤ 0.4; HoverAviary.py:110-113) **and** its altitude error |z − 1| ≤ **0.06 m**. A crash
(truncation before step 242) counts 0 for the remaining steps; a termination on reaching the target exactly
(:93-94) counts the remaining steps as successes.
- *Why 0.06 m:* it is the radius of the CF2X collision cylinder (`assets/cf2x.urdf:34`, `radius=".06"`, read as
  `COLLISION_R` at BaseAviary.py:1001): the drone "is at" the target height when the target lies within the
  airframe's own extent. Chosen before any `hover_success` number was seen.
- *Why altitude only:* `one_d_rpm` commands one collective thrust for all four motors (BaseRLAviary.py:223-224);
  with zero initial attitude no policy can move the drone horizontally, so the item's horizontal offset (up to
  0.35 m) is fixed by the item. Scored in 3-D, 87 % of items (offset > 0.06 m) would be 0 for every policy
  (`success_3d_frac` is kept as a diagnostic sub-metric and confirms it: 0.97 of items at 0 even for the best
  candidate).
- *Why the second half:* the first half is the transient from the item's start height and velocity; the second
  half measures holding the hover.

**Supporting metric `hover_score`** (evaluator `drones.sb3_return_items@0.1`, **T0**: it is the training reward
scaled, so under 05 §6.1 it never decides): episode return / (2 × 242), crash steps 0, a termination credited 2 per
remaining step. Both metrics are written per item: `--metric hover_success` (default) or `--metric hover_score`
chooses the arm's `score`, and `sub_metrics` always holds `hover_success`, `hover_score`, `success_3d_frac`,
`crashed`, `steps`, `within_10cm_frac`, `final_dist`, `final_z_err`, `mean_dist`, `episode_return`. A supporting
arm is a second run of `eval_hover.py` with `--metric hover_score` on the same cut. Policy
`predict(deterministic=True)` (Gaussian mean); observations normalized with the cut's VecNormalize statistics frozen
(`training=False`, `norm_reward=False`). A crash is a score; a simulator or policy exception is an error record of
class `simulator_error` (MM-4).

**K = 1.** Policy and simulator are deterministic: two evaluations of one cut (16 and 4 worker processes, and a
single process on a prefix) gave byte-identical `items.jsonl`. `eval_seeds = [0]` is recorded and seeds
random/numpy/torch, but nothing draws from them.

**Power (pilot-derived, `hover_success`).** Sanity pilots on the 300 items (CPU): base cut 4 → continue
cut 54: Δ̂ 0.784, sd of paired differences 0.341; base cut 8 → continue cut 58: Δ̂ 0.670, sd_diff 0.388. With
α = 0.05 one-sided and power 0.8 (z sum 2.487) the minimum detectable Δ at N = 300 is 2.487 · σ / √300: 0.049 for
σ 0.341 and 0.056 for σ 0.388 (0.063 at α = 0.025, the spec §6 default). Proposed registration: **σ = 0.39**
(`sigma_plan` from the pilot, the larger of the two), **margin δ = 0 (superiority)**, N = 300, K = 1, both marked
pilot-derived; the registered pair is base cut 8 → candidate (chosen 2026-11-12, see "Baseline cut change"), whose pilot
sd_diff 0.388 σ = 0.39 covers. `hover_score` (supporting): sd_diff 0.120 (cut 4 → 54) and 0.006 (cut 8 → 58).

**Disjoint training and evaluation states.** Training initial states come from each env's `np_random`, seeded once
with `seed + i` (gymnasium `SeedSequence(int)`), PPO seed 1000 for the base run and 2000 for the continue run;
items come from `SeedSequence([0x5EED17E5, item_seed])`, a different entropy. That is the design argument; the
check is exact: the item payload strings against every training episode's `payload` in `rollouts.jsonl`
(the analogue of MM-11's exact-text overlap). Sanity runs: 1,895 (base 4, smoke, full) and 2,029 (base 8 + full continuation) distinct training states,
**exact_overlap 0** in both.

## Seeds (MM-29)

`PPO(seed=s)` and `PPO.load(..., seed=s)` call `set_random_seed(s)` in `_setup_model`
(common/on_policy_algorithm.py:115-117): `random`, `numpy` and `torch` seeded (common/utils.py:28-40) and
`VecEnv.seed(s)`, which sets env i's next-reset seed to `s + i` (common/vec_env/base_vec_env.py:308) and clears it
after that reset (:96-100; dummy_vec_env.py:78-81, subproc_vec_env.py:142-148). Automatic resets pass no seed
(dummy_vec_env.py:71, subproc_vec_env.py:42), so each env's initial states are one `np_random` stream. `PPO.load`
applies keyword arguments before `_setup_model` (base_class.py:738-740), so `seed=2000` replaces the stored base
seed and the continue run does not replay the base run's initial states. On CPU with fixed thread counts the runs
are reproducible: two base runs gave identical `policy.pth`, `vecnormalize.pkl` and rollout logs (`model.zip`
differs only in its `data` timestamps and system info).

## SB3 facts (stable-baselines3 2.9.2, cited from the installed package)

- **Save format.** `save` (common/base_class.py:819-867) pickles nothing itself: `data` is `self.__dict__` minus
  `policy, device, env, replay_buffer, rollout_buffer, _vec_normalize_env, _logger, _custom_logger` (:303-321) and
  the torch state names, serialized to JSON with non-JSON values as base64 cloudpickle (save_util.py:76-100). The
  zip is written in this order (save_util.py:319-333): `data`, `pytorch_variables.pth`, one `<name>.pth` per state
  dict (PPO: `policy.pth`, `policy.optimizer.pth`, on_policy_algorithm.py:343-346), `_stable_baselines3_version`,
  `system_info.txt`; members are stored uncompressed. A 10-iteration cut: 15 kB `data`, 53 kB `policy.pth`,
  107 kB `policy.optimizer.pth`.
- **Load.** `PPO.load(path, env, **kwargs)` (base_class.py:643-800) rebuilds the model from `data`, sets
  `_last_obs = None` when an env is given (force_reset), updates `__dict__` with `data` then `kwargs`, calls
  `_setup_model`, then `set_parameters` loads `policy` and `policy.optimizer` (an optimizer's state dict is loaded
  as is, :614-630): **the Adam moments continue**. `num_timesteps`, `_n_updates`, `seed`, `n_steps`, `batch_size`,
  `n_epochs` come from `data`.
- **Continuing.** `learn(reset_num_timesteps=False)` keeps `num_timesteps` and `_episode_num` and adds it to the
  requested total (base_class.py:411-418); the env is reset because `_last_obs` is None (:421-425). Measured: base
  cut 32,768 steps / 40 updates, continue cut 114,688 / 140. The local `iteration` counter of `learn` restarts at 0
  (on_policy_algorithm.py:309); the cut names below use a global count.
- **Learn loop.** `while num_timesteps < total`: `collect_rollouts` (callbacks `on_rollout_start`, `on_step` after
  every `env.step`, `on_rollout_end`), then `train()` on that rollout (on_policy_algorithm.py:323-337; :192, :223-224,
  :264-266). `num_timesteps += n_envs` per step (:220).
- **CheckpointCallback** (common/callbacks.py:245-321) saves every `save_freq` `on_step` calls, that is every
  `save_freq × n_envs` env steps, **in the middle of a rollout** (:300-303): `<prefix>_<num_timesteps>_steps.zip`,
  then `<prefix>_vecnormalize_<num_timesteps>_steps.pkl` (:298, :314-317), flat files, no completeness marker. Not
  used: `train_ppo.py` saves its own cuts at rollout boundaries (below).
- **VecNormalize** `save` is `pickle.dump(self)` without the venv (vec_normalize.py:128-138, :324-332); `load`
  unpickles (:311-320). The adapter must hash it, never open it; `vecnormalize.json` mirrors its statistics.

## Outputs of `train_ppo.py` (`/out`)

| File | Written | Purpose |
|---|---|---|
| `env-freeze.txt` | first, before torch is imported | MM-33: sorted `name==version`, plus `# gym-pybullet-drones commit <sha>` (PEP 610 `direct_url.json`) |
| `env-runtime.json` | first | Python, lock SHA-256, `script_sha256` (EXT-83), `env_class_sha256`, uid/gid, CPU count |
| `run_args.json` | before training | phase, profile, every PPO / VecNormalize value, `seed`, `env_seeds`, seed derivation, `env_params_file` + `env_params_sha256` + `env_version_sha256` (data binding), `base_checkpoint` + `base_cut` (continue), `first_iteration`, `last_iteration` |
| `rollouts.jsonl` | at each rollout's end, before the update that uses it, fsync | observed consumption log (below) |
| `memory.jsonl` | per iteration | seconds, peak RSS |
| `attempts.jsonl` | start and end | attempt history; a rerun after a preemption moves earlier files to `preempted-attempt-<k>/` and starts again |
| `checkpoint-<I>/` | after update I | the cuts |

**Cut layout.** `checkpoint-<I>/`, I = global PPO iteration (base iterations 1-8, continue iterations 9 …). Written
into `.checkpoint-<I>.partial/` and renamed atomically: `model.zip` (SB3 save: resume state), `policy.pth` (the
zip's `policy.pth` member copied out: the weights), `policy_meta.json` (policy class, net_arch, spaces, SB3 version),
`vecnormalize.pkl`, `vecnormalize.json`, `env_version.json`, then **`cut.json` last** (iteration, `num_timesteps`,
`n_updates`, phase, SHA-256 of every other file). Cuts are taken at `on_rollout_start` after update I and at
`on_training_end`, every `save_every` iterations of the run (so a continue run from base 8 with `save_every 10`
cuts at 18, 28, 38, 48, 58); about 250 kB each.

**Consumption log `rollouts.jsonl`** (MM-25, MM-27; written at run time by the callback, never reconstructed). One
line per iteration I: `iteration`, `phase`, `num_timesteps_first`, `num_timesteps_last` (cumulative env steps of the
rollout, 1-based, inclusive), `n_envs`, `n_steps`, `env_params_version`, `shard` (`hover-init-v1.jsonl`), `utc`,
and `episodes`: every episode with at least one step in rollout I, `{env, episode, seed, init, payload, steps}`
(`seed` = the env's root seed `seed + env`, `episode` = per-env counter). The callback refuses a rollout whose
logged steps differ from `n_steps × n_envs`. An episode that spans two rollouts appears in both with its steps in
each. Update I uses exactly rollout I; cut I holds the weights after update I.

## Mounts and command shape

| Host | Container | Mode |
|---|---|---|
| `<data-root>/python` | same | ro |
| `<runtime>` (this directory: scripts, lock, venv, `env-snapshot/`, `items/`) | same | ro |
| continue only: `<output-root>/sb3-ppo-drone-hover/<base run>/checkpoint-8` | `/inputs/base` | ro |
| evaluation only: the cut to score | `/inputs/cut` | ro |
| `<output-root>/sb3-ppo-drone-hover/<run>` (new, empty, owned by the launching user) | `/out` | rw |
| tmpfs | `/tmp` | `rw,exec,nosuid,nodev,size=8g` |

The snapshot and items are read inside the `<runtime>` mount (their own mounts would nest, EXT-94).
Environment names (all of them; none is a credential): `HF_HUB_OFFLINE=1` (required by the launcher, unused),
`HOME=/tmp/home`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, `XDG_CACHE_HOME=/tmp/cache`. Network `none`
(nothing is downloaded), user `1000:1000`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--shm-size=8g`
(launcher default; SubprocVecEnv and the evaluation pool use `forkserver`, whose socket lives in `/tmp`).

Request shapes (`ext-launch --adapter sb3_ppo`, `gpus: false`):

```
base:      argv ["<runtime>/.venv/bin/python",
                 "<runtime>/train_ppo.py",
                 "--phase", "base", "--profile", "full", "--out", "/out",
                 "--env-snapshot", "<runtime>/env-snapshot"]
           training_seed 1000, label mm-a2-drones-base, output <output-root>/sb3-ppo-drone-hover/base-1
continue:  argv [<python>, <train_ppo.py>, "--phase", "continue", "--profile", "smoke"|"full", "--out", "/out",
                 "--env-snapshot", "<runtime>/env-snapshot", "--base", "/inputs/base"]
           training_seed 2000, labels mm-a2-drones-smoke / mm-a2-drones-full
evaluate:  argv [<python>, "<runtime>/eval_hover.py",
                 "--checkpoint", "/inputs/cut", "--items", "<runtime>/items",
                 "--env-snapshot", "<runtime>/env-snapshot", "--arm-out", "/out",
                 "--eval-seed", "0", "--workers", "16", "--model-hash", "<H>", "--item-set-hash", "<H>",
                 "--metric", "hover_success"]      # a second run with "hover_score" gives the supporting arm
```

`framework`: `{name: "stable-baselines3", version: "2.9.2", commit: null}`; `runtime`: `{kind: "venv_in_container",
image: <area-1 digest>, venv_lock: "<runtime>/uv.lock"}`. argv[0] resolves to `python3.12` in the Python
mount (EXT-91).

## Profiles and expected cost

PPO settings (both phases, `COMMON` in `train_ppo.py`): 8 SubprocVecEnv envs × 1024 steps = 8192 env steps per
iteration, batch 256, 10 epochs, Adam lr 3e-4 constant, γ 0.99, λ 0.95, clip 0.2, no entropy bonus, vf 0.5,
grad clip 0.5, `net_arch [64, 64]`, VecNormalize obs and reward (clip 10, γ 0.99), torch 4 threads.

| Run | Iterations | Cuts | Env steps | Wall (measured) |
|---|---|---|---|---|
| base smoke | 2 | 1, 2 | 16,384 | 9 s |
| **base full** | 8 | 4, **8 = the base** (see "Baseline cut change") | 65,536 | 25 s |
| continue smoke | 12 | 12, 16, 20 | 98,304 | 27 s |
| continue full | 50 | 18, 28, 38, 48, 58 | 409,600 | 105 s |

Continue lengths (chosen 2026-11-12): smoke 10-15, full about 50, cuts every 10; the score saturates within about
10 iterations.

Host memory: about 0.35 GB per process, 9 processes in training (3-4 GB peak with headroom), 17 in evaluation with
16 workers; no swap expected; disk about 250 kB per cut, under 10 MB of logs. Evaluation: 300 items in about 5 s
with 16 workers.

***Baseline cut change (2026-11-12).** The base is **cut 8** of the base run
(8 iterations, 65,536 env steps), not cut 4. Reason: the cut-4 policy never leaves the ground (its 211 non-crashing
episodes all end 0.52-0.99 m below the target, median 0.987; 89 crash), so the deciding T3 metric is 0 on all
300 items and MM-38 would refuse it. Disclosure: the change was made before the registration and before any
result of the gated candidate run; it was informed by the CPU sanity pilots (the base learning curve and
pilot continuations, not published), and the tolerance 0.06 m was fixed from the collision radius
before any `hover_success` number was seen and was not changed.

**Base policy (cut 8).** `hover_success` 0.322 ± 0.390 (min 0, max 1), 49.0 % of items at 0, 11.3 % at 1;
`hover_score` 0.988 ± 0.014; no crashes. **Known risk (MM-38):** 49.0 % of items share the value 0, one point under
the 50 % rule; the container smoke must re-check it on the evaluated base cut (a numerically different container
run could cross the line). Note also that the supporting T0 `hover_score` baseline (0.988) sits in the top 5 % of
its range; it is non-blocking. Cut 4 for reference: `hover_success` 0 on every item at 0.06 m and at 0.10 m,
`hover_score` 0.595 ± 0.122.
Learning curve of the base run (300 items, iteration: `hover_success` mean / at 0 / `hover_score` mean): 4: 0.000 /
1.00 / 0.595; 5: 0.000 / 1.00 / 0.788; 6: 0.000 / 0.99 / 0.940; 7: 0.070 / 0.81 / 0.982; 8: 0.322 / 0.49 / 0.988;
10: 0.215 / 0.53 / 0.990.

## Sanity check (2026-11-12, host CPU, no container, outputs not published)

- `uv lock` 4 s, `uv sync --frozen` including the pybullet sdist build: about 1 min; imports OK (torch 2.15.0+cpu).
- Base full: 11-15 s; two runs gave identical weights, normalizer and rollout logs; a 10-iteration base run with the
  same seed reproduces cut 4 bit for bit; the final base profile (8 iterations) reproduced cut 8's `policy.pth`,
  `vecnormalize.pkl`/`.json` and `env_version.json` of an earlier 10-iteration run bit for bit (only `model.zip`
  timestamps differ), and its continuation reproduced cut 58's weights and normalizer. Continue smoke (12 iterations) 27 s, continue full (50) 106 s, about
  2.1 s per iteration; `num_timesteps` and `n_updates` continue across the phases.
- Evaluation, 300 items, 16 workers: about 5 s per arm; byte-identical across worker counts. 0 errors everywhere.

| Arm (300 items, tol 0.06 m) | `hover_success` mean / sd / min / max / at 0 / at 1 | `hover_score` mean / sd | crashes |
|---|---|---|---|
| base cut 4 (not used) | 0.000 / 0.000 / 0 / 0 / 1.000 / 0.000 | 0.595 / 0.122 | 89 |
| continue smoke cut 16 (from cut 4) | 0.004 / 0.030 / 0 / 0.377 / 0.977 / 0.000 | 0.990 / 0.010 | 0 |
| continue full cut 14 (from cut 4) | 0.028 / 0.108 / 0 / 0.967 / 0.890 / 0.000 | 0.990 / 0.011 | 0 |
| continue full cut 54 (from cut 4) | 0.784 / 0.341 / 0 / 1 / 0.117 / 0.493 | 0.993 / 0.008 | 0 |
| **base cut 8 (the base)** | 0.322 / 0.390 / 0 / 1 / 0.490 / 0.113 | 0.988 / 0.014 | 0 |
| **continue full cut 58 from cut 8 (candidate)** | 0.992 / 0.031 / 0.779 / 1 / 0.000 / 0.883 | 0.993 / 0.008 | 0 |

At 0.10 m (reported, not adopted): base cut 4 0.000 (all 0), base cut 8 0.504 ± 0.427 (at 0 0.33, at 1 0.28),
continue cut 54 0.983, continue cut 58 from 8 1.000.

## What gatedtrain needs (adapter `sb3_ppo`, trainer id `rl.sb3_ppo@0.1`, evaluator `drones.sb3_items@0.1`)

1. **Cuts and completeness (MM-20).** `checkpoint-(\d+)/` with `cut.json` whose `files` SHA-256 all match and whose
   `iteration` equals the directory number; `.checkpoint-*.partial/` and directories without `cut.json` are
   incomplete. The last cut's iteration equals `run_args.json` `last_iteration` (EXT-68 analogue).
2. **Identity (MM-17, MM-19).** `model_hash` components: `policy.pth`, `policy_meta.json`, `vecnormalize.pkl`,
   `vecnormalize.json`, `env_version.json`; `cut_hash` adds `model.zip` and `cut.json`. Readable without pickle:
   `cut.json`, `policy_meta.json`, `vecnormalize.json`, `env_version.json`, and `model.zip` members `data` (JSON;
   leave its base64 cloudpickle values undecoded) and `_stable_baselines3_version` (stdlib `zipfile`). The arm's
   `normalizer_sha256` is the SHA-256 of `vecnormalize.pkl`; `env_version_sha256` is the SHA-256 of
   `env-snapshot/env_version.json`, equal to each cut's copy.
3. **Base binding (MM-21).** Base run: no base model (trained from scratch, `base_checkpoint` null); its cut 8 is the
   gate's baseline. Continue run: `run_args.json` `base_checkpoint` is the container path of the read-only mount whose
   host path is the spec's base cut, and `base_cut.files` equals that cut's `cut.json` files (MM-15/MM-30 chain:
   the continue run's initial model hash is the base run's cut `model_hash`).
4. **Data binding (EXT-65 analogue).** Spec data = `env-snapshot/shards/`; `run_args.json` `env_params_file` lies in
   a read-only mount whose host side is that directory or contains it, and `env_params_sha256` equals the shard's
   SHA-256; `env_version_sha256` equals the registered environment's.
5. **Seed and pins.** `run_args.json` `seed` equals the launch record's `training_seed` (1000 base, 2000 continue);
   `env-freeze.txt` has `stable-baselines3==2.9.2` (the framework pin) and the gym-pybullet-drones commit line;
   `env-runtime.json` `script_sha256` equals the record's (EXT-83).
6. **Consumption (MM-25).** Declared (2026-11-12): precision `iteration`, **derivation
   `observed`**: one row per (iteration,
   shard) from `rollouts.jsonl`, `step` = global iteration, `chunk = {first: num_timesteps_first, last:
   num_timesteps_last}`, `seed_inputs.seed` = run seed, `data_ref.snapshot` = manifest hash of `env-snapshot/`,
   `shard` = `hover-init-v1.jsonl`, `row` null; only iterations ≤ the last cut (later ones orphaned, MM-27).
   Refuse when the log's iterations are not consecutive from `first_iteration`, or its steps per line differ from
   `n_steps × n_envs`. The per-episode lines stay in the log (exact-overlap check, later use) but `row` is not
   claimed. SB3's own logger (`time/total_timesteps`) is not needed.
7. **Exact overlap (MM-11).** Item `payload` strings against every `episodes[].payload` of the training runs'
   `rollouts.jsonl` (registry `payload_text_field: "payload"`), not against the shard text.
8. **Registry rows.** Trainer: `framework: "stable-baselines3"`, `pins: ("stable-baselines3", "gymnasium",
   "gym-pybullet-drones", "pybullet")`; `input_roots`: the Python directory, `<runtime>`,
   and the base run's output (`<output-root>/sb3-ppo-drone-hover`, read-only for the continue run);
   `env_names`: `HF_HUB_OFFLINE`, `HOME`, `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`, `XDG_CACHE_HOME`;
   `scan_skip`: `<runtime>/.venv` (36,869 entries; no credential-like names found)
   and the interpreter's site-packages; `interpreter_trees`: the CPython 3.12.16 tree. Evaluators, both
   `higher_is_better`, `(0, 1)`, not binary, `error_classes ("simulator_error",)`, `score_outcomes ("crash",
   "below_tolerance", "timeout")`, `normalizer: True`, `env_version: True`, `item_payload ("init", "payload")`,
   `payload_text_field: "payload"`, samples "300, 1":
   `drones.sb3_items@0.1` metric `hover_success`, **tier T3**, deciding, scorer `{name: hover_success, version:
   eval_hover@0.2, weights_sha256: null, code_sha256: SHA-256 of eval_hover.py}`;
   `drones.sb3_return_items@0.1` metric `hover_score`, **tier T0** (training reward, non-blocking), scorer
   `{name: hover_score, version: eval_hover@0.2, ...}`.
9. **Arm building.** `eval_hover.py` writes an MM-1/MM-2 arm in one piece (no `merge.json`, provenance is the
   evaluator's claim, MM-5): `items.jsonl` lines `{item, repeat 0, seed, score, error, sub_metrics, output_ref null,
   item_meta {init, env_params_version}}`, `eval_meta.json` with every MM-2 key, plus `summary.json` and
   `env-runtime.json` (not read by the gate). It cannot compute gatedtrain's `model_hash` or the item set manifest
   hash: pass them as `--model-hash` / `--item-set-hash` (the adapter's `identity(cut)` and `manifest_hash(items/)`),
   or add a small generic `ext-arm-fill` step that rewrites `checkpoint.model_hash`, `item_set_hash` and
   `sampler_hash` (here SHA-256 of compact sorted JSON, not RFC 8785) in `eval_meta.json`. `ext-from-nel` does not
   apply (no NeMo Evaluator runs).

## Deviations from MULTIMODAL-SPEC §6 row 4 (accepted 2026-11-12)

- Metric: §6 names `episode return, [null, null]`; this runtime decides on the bounded T3 `hover_success`
  ([0, 1]) and reports the return as the bounded T0 `hover_score` ([0, 1]), a supporting evaluator.
- Items: N = 300 (the table's estimate is 200), K = 1.
- The request's `area` integer is 4 (decision and spec row 4), though the original runtime directory was named
  `area2-drones` after the run order.

## Decisions taken (2026-11-12)

- Deciding metric T3 `hover_success`; `hover_score` supporting (T0).
- Continue length: smoke 12 iterations, full 50, cuts every 10 (every 4 in smoke).
- GPU accounting: `gpus: false` in the request; ext-collect logs 0 GPU-hours for CPU steps.
- Consumption: `iteration`, observed; per-episode rows logged, `row` not claimed.
- Base = cut 8 of the base run; tolerance 0.06 m unchanged; σ 0.39, δ 0, N 300, K 1 (pilot-derived).

## Open questions

1. **MM-38 margin of the base (known risk).** Resolved by taking base cut 8 (tolerance unchanged), but 49.0 %
   of base items score 0 against the 50 % line; re-check in the container smoke before the full run. Pilot
   numbers for the registered pair (cut 8 → cut 58): Δ̂ 0.670, sd_diff 0.388, 88.7 % of items better, 0.3 % worse.
2. `HF_HUB_OFFLINE=1` is required by the launcher but meaningless here; kept to satisfy the rule.
3. Changing `HoverItemEnv` changes `env_class_sha256`; the env snapshot must then be regenerated
   (`train_ppo.py --write-env-snapshot`), and both scripts refuse a mismatch.
