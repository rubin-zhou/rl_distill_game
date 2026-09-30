"""PPO training for Flappy Bird. Run inside mlenv:
    conda activate mlenv
    python train/train_ppo.py                          # pixels, 1M steps, 16 envs
    python train/train_ppo.py --obs vector --timesteps 200000
    python train/train_ppo.py --resume runs/xxx/ppo_flappy.zip
"""
import os
import sys
import argparse
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import (CallbackList, CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env.flappy_env import FlappyEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", choices=["pixels", "vector", "dict"],
                   default="pixels", help="dict = 84x84 image + 7d vector sidecar "
                   "(MultiInputPolicy)")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=16,
                   help="parallel envs; total_timesteps counts env steps summed")
    p.add_argument("--gap", type=float, default=110.0)
    p.add_argument("--shaping", type=int, default=1)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--exp-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                   help="cpu = 显存零占用, 适合与本地 LLM 共存 (推荐)")
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn"],
                   help="dqn = ε-随机探索, 对像素 flappy 收敛更快")
    return p.parse_args()


def make_env_fn(kwargs):
    def fn():
        return FlappyEnv(**kwargs)
    return fn


def main():
    args = parse_args()
    exp = args.exp_name or f"flappy_{args.obs}_g{int(args.gap)}"
    run_dir = ROOT / "runs" / exp
    log_dir = run_dir / "logs"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(obs_mode=args.obs, gap_height=args.gap,
                      shaping=bool(args.shaping))
    vec = SubprocVecEnv([make_env_fn(env_kwargs) for _ in range(args.n_envs)])
    if args.obs == "vector" and args.algo == "ppo":
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=20.0)

    single = DummyVecEnv([
        lambda: Monitor(FlappyEnv(**env_kwargs), str(run_dir / "eval_logs"))])
    if args.obs == "vector" and args.algo == "ppo":
        single = VecNormalize(single, norm_obs=False, norm_reward=False, training=False)

    pixels = (args.obs in ("pixels", "dict"))
    policy = ("MultiInputPolicy" if args.obs == "dict" else
              "CnnPolicy" if args.obs == "pixels" else "MlpPolicy")
    if args.algo == "dqn":
        model_args = dict(
            policy=policy, env=vec,
            learning_rate=args.lr or 1e-4,
            buffer_size=150_000, batch_size=128, learning_starts=8000,
            train_freq=(1, "step"), gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.15, exploration_final_eps=0.04,
            gamma=0.99, verbose=1, seed=1,
            tensorboard_log=str(log_dir),
        )
    else:
        model_args = dict(
            policy=policy,
            env=vec,
            learning_rate=args.lr or (1e-4 if pixels else 3e-4),
            n_steps=512 if pixels else 256,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            verbose=1,
            seed=1,
            tensorboard_log=str(log_dir),
        )
    if args.device == "cpu":
        # cap torch threads so the game-logic worker processes get cores (GB10/20-core box)
        import torch as _t
        _t.set_num_threads(int(os.environ.get("TRAIN_THREADS", "6")))
    if args.device in ("cpu", "cuda"):
        model_args["device"] = args.device
    Cls = DQN if args.algo == "dqn" else PPO
    load_kw = dict(env=vec, tensorboard_log=str(log_dir),
                   device=args.device if args.device in ("cpu", "cuda") else "auto")
    model = Cls.load(args.resume, **load_kw) if args.resume else Cls(**model_args)

    eval_cbs = EvalCallback(single, best_model_save_path=str(run_dir / "eval"),
                            eval_freq=max(131072 // args.n_envs, 16),
                            n_eval_episodes=10, deterministic=True)
    ckpt = CheckpointCallback(save_freq=max(262144 // args.n_envs, 16),
                              save_path=str(run_dir / "checkpoints"),
                              name_prefix="ppo")
    model.learn(total_timesteps=args.timesteps,
                callback=CallbackList([ckpt, eval_cbs]))

    model.save(str(run_dir / "ppo_flappy"))
    if hasattr(vec, "save"):
        vec.save(str(run_dir / "vecnormalize.pkl"))
    print("saved:", run_dir / "ppo_flappy.zip")


if __name__ == "__main__":
    main()
