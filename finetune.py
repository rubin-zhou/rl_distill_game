"""BC -> PPO fine-tune: load distilled conv weights as the features extractor,
then PPO continues training the pixel policy directly (this breaks the teacher's
ceiling; best run on GPU).

Usage (mlenv, from repo root, GPU mode):
    python finetune.py --init runs/distill/dagger.pt --timesteps 3000000 --n-envs 16

For coexistence with the local LLM use --device cpu (runs ~2x slower but zero
GPU use).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
ROOT = pathlib.Path(__file__).resolve().parent

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env.flappy_env import FlappyEnv
from sb3_io import load_ppo


class FlappyConv(BaseFeaturesExtractor):
    """conv trunk identical to distill.PixelCnn, returning flattened features"""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=64 * 7 * 7)
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )

    def forward(self, obs):
        x = obs.to(torch.float32)
        if x.shape[-1] in (1, 3) and x.shape[-1] != x.shape[1]:
            x = x.permute(0, 3, 1, 2)            # (B,H,W,C) -> (B,C,H,W)
        x = x / 255.0 if x.max() > 6 else x      # uint8-normalized or already float
        h = self.body(x)
        return h.flatten(1)


def load_conv_weights(model, pt_path):
    """copy conv trunk AND the action head from a distill.PixelCnn checkpoint.

    The student head is Linear(3136->256)+ReLU+Linear(256->2); with
    net_arch=dict(pi=[256]) + activation_fn=ReLU the SB3 policy becomes
    Linear(3136->256)+ReLU+Linear(256->2) as well, so the whole policy
    (not just the conv) starts from the working student -> PPO fine-tune
    begins at the student's ~50 score instead of a random head (~0).
    """
    from distill import PixelCnn
    sd = torch.load(pt_path, map_location="cpu")["state"]
    src = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
    feat = model.policy.features_extractor
    mapping = {"0.weight": src.get("0.weight"), "0.bias": src.get("0.bias"),
               "2.weight": src.get("2.weight"), "2.bias": src.get("2.bias"),
               "4.weight": src.get("4.weight"), "4.bias": src.get("4.bias")}
    dst = dict(feat.body.named_parameters())
    ok = 0
    for name, w in mapping.items():
        if w is not None and name in dst:
            with torch.no_grad():
                dst[name].copy_(w)
            ok += 1
    head_ok = 0
    pol = model.policy.mlp_extractor.policy_net
    an = model.policy.action_net
    with torch.no_grad():
        pol[0].weight.copy_(sd["net.7.weight"]); pol[0].bias.copy_(sd["net.7.bias"])
        an.weight.copy_(sd["net.9.weight"]);     an.bias.copy_(sd["net.9.bias"])
    head_ok = 4
    print(f"loaded {ok}/6 conv tensors + {head_ok}/4 head tensors from {pt_path}",
          flush=True)

    # sanity: policy logits must equal student logits (same function now)
    x = (torch.rand(8, 84, 84, 1) * 255.0).to(next(model.policy.parameters()).device)
    student = PixelCnn(); student.load_state_dict(sd)
    student.eval().to(x.device)
    model.policy.eval()
    with torch.no_grad():
        f = model.policy.features_extractor(x)
        lat = model.policy.mlp_extractor.forward(f)[0]
        ours = model.policy.action_net(lat)
        theirs = student(x / 255.0)             # student permutes HWC->CHW itself
    diff = (ours - theirs).abs().max().item()
    print(f"logit max|diff| vs student: {diff:.6f} "
          f"({'OK: init policy == student' if diff < 1e-4 else 'MISMATCH!'})",
          flush=True)
    model.policy.train()

    # zero-init value HEAD only (keep trunk random: a fully-zero trunk dies
    # behind ReLU(0) with zero gradient -> value learns nothing, EV stays 0,
    # advantages keep no long-range info and PPO collapses the policy).
    with torch.no_grad():
        model.policy.value_net.weight.zero_()
        model.policy.value_net.bias.zero_()
    print("value head zero-init done (trunk kept random)", flush=True)


def make_env_fn(kwargs):
    def fn():
        return FlappyEnv(**kwargs)
    return fn


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="runs/distill/dagger_v1.pt")
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--gap", type=float, default=110.0)
    p.add_argument("--lr", type=float, default=5e-5,
                   help="small: we start from a working policy, only adjust")
    p.add_argument("--exp-name", default="run8_ft")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--resume", default=None)
    p.add_argument("--warmup", type=int, default=200_000,
                   help="freeze policy (lr=0), train value net only, N steps")
    a = p.parse_args()

    run_dir = ROOT / "runs" / a.exp_name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(obs_mode="pixels", gap_height=a.gap, shaping=True)
    venv = SubprocVecEnv([make_env_fn(env_kwargs) for _ in range(a.n_envs)])
    eval_env = DummyVecEnv([lambda: Monitor(FlappyEnv(**env_kwargs),
                                            str(run_dir / "eval_logs"))])

    # value-warmup gate: with a cold value net the advantages carry no
    # long-range info and PPO collapses the good student policy (observed
    # 7500 -> 255 -> 62). Freeze the POLICY via requires_grad for N steps so
    # only the value net learns; then unfreeze for small-step updates.
    # (A separate optimizer group would NOT work: SB3's lr schedule writes
    #  the same lr into every param_group.)
    warmup = {"frozen": (not a.resume) and a.warmup > 0}
    pol_params = None

    model = (load_ppo(a.resume, env=venv) if a.resume else
             PPO("CnnPolicy", venv,
                 policy_kwargs=dict(features_extractor_class=FlappyConv,
                                    net_arch=dict(pi=[256], vf=[128]),
                                    activation_fn=nn.ReLU),
                 learning_rate=lambda frac: a.lr * frac,
                 n_steps=512, batch_size=128, n_epochs=5,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=0.0,
                 verbose=1, seed=1, device=a.device,
                 tensorboard_log=str(run_dir / "logs")))
    if not a.resume:
        load_conv_weights(model, a.init)
        pol_params = (list(model.policy.features_extractor.parameters())
                      + list(model.policy.mlp_extractor.policy_net.parameters())
                      + list(model.policy.action_net.parameters()))
        if warmup["frozen"]:
            for p in pol_params:
                p.requires_grad_(False)
            print(f"policy frozen for warmup "
                  f"({len(pol_params)} tensors, requires_grad=False)",
                  flush=True)

    class Unfreeze(BaseCallback):
        def _on_step(self):
            if warmup["frozen"] and self.num_timesteps >= a.warmup:
                warmup["frozen"] = False
                for p in pol_params:
                    p.requires_grad_(True)
                ev = self.logger.name_to_value.get("train/explained_variance")
                print(f"\npolicy unfrozen at {self.num_timesteps} steps, "
                      f"value explained_variance={ev}", flush=True)
            return True

    eval_cbs = EvalCallback(eval_env, best_model_save_path=str(run_dir / "eval"),
                            eval_freq=max(196608 // a.n_envs, 16),
                            n_eval_episodes=10, deterministic=True)
    ckpt = CheckpointCallback(save_freq=max(393216 // a.n_envs, 16),
                              save_path=str(run_dir / "checkpoints"),
                              name_prefix="ft")
    model.learn(total_timesteps=a.timesteps,
                callback=CallbackList([Unfreeze(), ckpt, eval_cbs]))
    model.save(str(run_dir / "ft_flappy"))
    print("saved:", run_dir / "ft_flappy.zip")
