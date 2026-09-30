import sys, glob, os, re
sys.path.insert(0, ".")
import numpy as np
from sb3_io import load_ppo
from env.flappy_env import FlappyEnv

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="runs/run4_dict")
    p.add_argument("--obs", default="dict", choices=["pixels", "vector", "dict"])
    p.add_argument("--episodes", type=int, default=10)
    a = p.parse_args()
    bm = os.path.join(a.run, "eval", "best_model.zip")
    ck = glob.glob(os.path.join(a.run, "checkpoints", "*.zip"))
    if os.path.exists(bm):
        f = bm
    elif ck:
        f = max(ck, key=lambda s: int(m.group(1)) if (m := re.search(r"(\d+)_steps", s)) else -1)
    else:
        sys.exit(f"no model found under {a.run}")
    m = load_ppo(f)
    env = FlappyEnv(obs_mode=a.obs, shaping=False)
    sc = []
    for _ in range(a.episodes):
        o, _ = env.reset(); d = False; t = 0
        while not d and t < 30000:      # cap high: 5000 hid real strength
                                         # (71 = 5000/71-frames-per-pipe = cap)
            a, _ = m.predict(o, deterministic=True)
            o, r, d, tr, i = env.step(int(np.asarray(a).item())); t += 1
        sc.append(env.state.score)
    print(f, "scores:", sc, "avg", round(sum(sc) / len(sc), 2))
