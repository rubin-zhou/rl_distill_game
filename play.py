"""Watch a trained model play. Run inside mlenv:
    conda activate mlenv
    python play.py                                # default model, on-screen
    python play.py runs/xxx/eval/best_model.zip    # specific model
    python play.py --headless --episodes 15       # record mp4s to videos/
    python play.py --obs vector                   # MLP/vector model
"""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    import numpy as np
    import argparse
    p = argparse.ArgumentParser()
    import sys
    p.add_argument("model", nargs="?", default=None)
    p.add_argument("--obs", choices=["pixels", "vector", "dict"], default="pixels")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video-dir", default=str(ROOT / "videos"))
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--gap", type=float, default=110.0)
    args = p.parse_args()

    def load_any(path):
        if path.endswith(".pt"):      # distilled CNN
            from distill import DistilledPolicy
            return DistilledPolicy(path)
        from sb3_io import load_ppo
        return load_ppo(path)

    model_path = args.model or str(ROOT / "runs/run2/eval/best_model.zip")
    from env.flappy_env import FlappyEnv

    if args.headless:
        import cv2
        from game.core import WIDTH, HEIGHT
        os.makedirs(args.video_dir, exist_ok=True)
        env = FlappyEnv(obs_mode=args.obs, gap_height=args.gap, shaping=False)
        model = load_any(model_path)
        scores = []
        for ep in range(1, args.episodes + 1):
            writer = cv2.VideoWriter(
                os.path.join(args.video_dir, f"ep{ep}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 60, (WIDTH, HEIGHT))
            obs, info = env.reset()
            done, step = False, 0
            writer.write(cv2.cvtColor(env.render("rgb_array"), cv2.COLOR_RGB2BGR))
            while not done and step < 20000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(int(np.asarray(action).item()))
                done = term or trunc
                writer.write(cv2.cvtColor(env.render("rgb_array"), cv2.COLOR_RGB2BGR))
                step += 1
            writer.release()
            scores.append(env.state.score)
            print(f"episode {ep}: score={scores[-1]}")
        print("avg score:", sum(scores) / len(scores), "videos ->", args.video_dir)
    else:
        import pygame
        env = FlappyEnv(obs_mode=args.obs, gap_height=args.gap, shaping=False)
        model = load_any(model_path)
        pygame.init()
        clock = pygame.time.Clock()
        scores = []
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done, step = False, 0
            while not done and step < 20000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = env.step(int(np.asarray(action).item()))
                done = term or trunc
                env.render("human")
                if env.quit_requested:
                    pygame.quit()
                    return
                clock.tick(60)
                step += 1
            scores.append(env.state.score)
            print(f"episode {ep + 1}: score={scores[-1]}")
        print("avg:", sum(scores) / len(scores))


if __name__ == "__main__":
    main()
