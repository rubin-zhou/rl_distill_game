# rl_distill-game — Training a Pixel-Input Flappy Bird AI by Teacher–Student Distillation

## Intro

Pixel-fed reinforcement learning cannot learn Flappy Bird on a budget: pure
pixel PPO or DQN trained for 3M steps still scores 0–2. The problem is not the
algorithm but the input — a CNN gets a very weak gradient for deciding *when
to flap*.

This repo sidesteps that with a teacher–student pipeline: train an expert in a
compact vector state space (where learning is fast), **distill its policy into
a pixel CNN via DAgger** (the student flies on its own, the teacher labels the
states the student actually visits — this fixes behavior-cloning's compounding
error), then **fine-tune with PPO** using a safe-init + value-warmup scheme
that avoids the policy collapse BC→PPO fine-tuning usually causes.

Everything runs on a single modern machine with one GPU.

<p align="center">
  <img src="video/learning_progress.gif" alt="Training progression: direct pixel
  PPO (score 0) → vector teacher (2 pts) → distilled pixel student (236 avg)" width="340">
  <br><sub>The product: three stages of the run played back to back — direct
  pixel PPO dies at the first pipe (0 pts), early vector teacher (2 pts), and
  the distilled pixel student flying (236 avg).</sub>
</p>

## Setup

Python 3.12, tested in a conda env (`mlenv`). Installs only 4 new packages
and leaves existing torch/numpy untouched:

```bash
conda create -n mlenv python=3.12 -y      # or any py3.12 env
pip install -r requirements.txt
# adds: pygame, gymnasium, stable-baselines3, tensorboard
# expects: torch (any CUDA build), numpy already installed
```

Environment self-check (~10 s, prints `OK`):
```bash
python -c "import sys; sys.path.insert(0,'.'); from env.flappy_env import FlappyEnv; from stable_baselines3.common.env_checker import check_env; check_env(FlappyEnv()); print('OK')"
```

Repo files: `game/` (game core + renderer), `env/flappy_env.py` (Gymnasium
env), `train/train_ppo.py`, `distill.py`, `dagger.py`, `finetune.py` (training
stages), `play.py` / `eval_scores.py` / `score_final.py` (evaluate & demos),
`sb3_io.py` (checkpoint-load workaround). `runs/` and `videos/` are gitignored.

## Usage Guide

The pipeline is four one-command steps. Each step's output feeds the next:

```
Step 1-2: vector expert (PPO, 7-dim state)     fast, reliable
      │  teacher labels the states the student visits
Step 3: DAgger distillation → pixel CNN         student ≈ expert
      │  conservative PPO fine-tune
Step 4: fine-tuned pixel policy                 can exceed the teacher
```

| Step | Produces | What to expect (our run) |
|---|---|---|
| 1. warmup expert | `runs/quick` | 3 min, learns basics (2–6) |
| 2. train teacher | `runs/vec10m/eval/best_model.zip` | ~1 h, avg ≥ 30 |
| 3. DAgger distill | `runs/distill/dagger_v1.pt` | ~25 min CPU, **avg 236/game** (75/433/145/291) |
| 4. PPO fine-tune | `runs/run8_ft/eval/best_model.zip` | ~30 min GPU, comparable to student |

For reference, training the same pixel CNN directly (Steps skip) never made
it past 13 in our experiments.

### Game
```bash
python game/flappy.py     # Space=flap, R=restart, Esc=quit
```

### Step 1: Warmup the vector teacher (3 min, optional)

Pipeline proof — vector-state learning is dramatically faster than pixel learning:

```bash
python train/train_ppo.py --obs vector --timesteps 300000 --n-envs 8 \
       --device cpu --exp-name quick
python eval_scores.py runs/quick --obs vector            # true score report
```

### Step 2: Train the real teacher (~1 h)

A stronger teacher directly means a stronger student, but you can stop as soon
as the eval score clears ~30 — no need to run the full 10M steps:

```bash
python train/train_ppo.py --obs vector --timesteps 10000000 --n-envs 16 \
       --exp-name vec10m
grep "Eval num" runs/vec10m.log | tail -1    # monitor growth
```

### Step 3: DAgger distillation (CPU, ~25 min, idempotent)

The student flies and collects states; the expert labels them; the student
retrains on that. Three rounds, each stronger than the last (covariate-shift
correction is why plain cloning only reached avg 5 here):

```bash
python dagger.py --expert runs/vec10m/eval/best_model.zip \
                 --init runs/distill/pixel_cnn_v2.pt \
                 --out runs/distill/dagger_v1.pt \
                 --rounds 3 --steps 150000 --device cpu
```
Each round prints `round N eval scores ... avg` — expect a strong upward trend.

### Step 4: Pixel PPO fine-tune (GPU, ~30 min)

Starts from the student weights and adds small on-policy updates to try to
pass the teacher. This step is delicate — the startup self-check prints four
validation lines, and if any is missing the run will collapse (all failure
modes and fixes are documented in the code):

```bash
python finetune.py --init runs/distill/dagger_v1.pt \
                   --timesteps 3000000 --n-envs 16 --device cuda --exp-name run8_ft
```

```
logit max|diff| vs student: 0.000000 (OK: init policy == student)
value head zero-init done (trunk kept random)
policy frozen for warmup (10 tensors, requires_grad=False)
policy unfrozen at 200000 steps, value explained_variance=0.5+
```

### Resume anything

```bash
python finetune.py --resume runs/run8_ft/checkpoints/$(ls -t runs/run8_ft/checkpoints | head -1) \
       --init runs/distill/dagger_v1.pt     # train_ppo.py accepts --resume the same way
```

### Scoring and demos

```bash
python eval_scores.py runs/run8_ft --obs pixels --episodes 10   # true avg score
python play.py runs/run8_ft/eval/best_model.zip                 # watch it fly
python play.py runs/distill/dagger_v1.pt --obs dict --headless --episodes 5
# demo videos are saved under videos/
```

## Reward & Environment Design

- `+0.5` per survive-frame, `+20` per pipe passed, `-20` crash.
- Optional potential-based shaping: only active within 180 px of the next
  pipe — idle circling cannot farm reward.
- Pixel obs = 400×600 scene downsampled to 84×84 grayscale; vector obs = 7
  normalized state values.
- Difficulty knob `--gap 60~110` (corridor width).
- `--obs dict` (pixels ⊕ vector side-channel) exists because we measured pure
  pixel CNN learning to be very slow; the side-channel fixed it. Observation
  design mattered more than algorithm choice.

## Gotchas (the reason this README exists)

1. **Frame caps fake your score.** Eval loops default to an episode frame
   cap; since you pass a pipe every ~71 frames, a score that is *exactly*
   `cap/71` means **survived the whole cap**, not death. All eval tools now
   use a 30000-frame cap; trust `eval_scores.py` only.
2. **Class imbalance kills naive cloning.** The expert flaps in only ~6% of
   frames. Plain cross-entropy converges to a 94%-accurate policy that never
   jumps. Category weighting (~17:1) is what makes distillation work.
3. **BC→PPO fine-tuning collapses quantitatively, and the fixes are in the
   code** (see `finetune.py`):
   | version | failure | fix in code |
   |---|---|---|
   | v1 | loaded only the conv trunk: head is random, start = random policy, scores 0 | load conv *and* full policy head (`net_arch pi=[256], ReLU`) |
   | v2 | random value net poisons advantage signs | zero-init value *head* only |
   | v3 | zero-init whole value trunk: `ReLU(0)` dead-gradient, value never learns | keep value trunk random |
   | v4 | freezing policy via a 2nd optimizer group: SB3's lr schedule writes **every** param group, freezing value too | freeze via `requires_grad=False` |
   | v5 | stable | 200k-step value warmup then small-lr updates |
4. **Reading training logs:** `explained_variance == 0` for many iterations
   means the value net is dead — collapse is guaranteed. eval reward ≈
   `0.5*frames + 20*score − 20`: use it for trend only. Early deaths in every
   episode at `l≈97` frames = broken action head. `entropy` crash = policy
   collapse → raise `ent_coef` to 0.05 or halve lr.

## Notes

- If your GPU is shared with another large workload, free the device (or
  just add `--device cpu` for steps 2–3; only step 4 really wants CUDA).
- `runs/` checkpoints from early runs are safe to delete if you need space.
