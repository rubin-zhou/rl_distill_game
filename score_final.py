import sys
sys.path.insert(0, ".")
import numpy as np
from sb3_io import load_ppo
from env.flappy_env import FlappyEnv

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "runs/run4_dict/ppo_flappy.zip"
    m = load_ppo(model)
    import gymnasium as gym
    from gymnasium import spaces
    os_ = m.observation_space
    if isinstance(os_, spaces.Dict):
        mode = "dict"
    elif os_.shape == (7,):
        mode = "vector"
    else:
        mode = "pixels"
    env = FlappyEnv(obs_mode=mode, shaping=False)
    sc = []
    for _ in range(10):
        o, _ = env.reset(); d = False; t = 0
        while not d and t < 8000:
            a, _ = m.predict(o, deterministic=True)
            o, r, d, tr, i = env.step(int(np.asarray(a).item())); t += 1
        sc.append(env.state.score)
    print(f"model: {model} [obs={mode}]")
    print("scores:", sc, " avg:", round(sum(sc)/len(sc), 2), " max:", max(sc))
