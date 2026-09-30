"""DAgger distillation: the pixel student collects states itself, the vector
expert labels the actions, then the student trains on that. Fixes BC's
covariate-shift/compounding-error problem.

Usage (mlenv, from repo root):
    python dagger.py --expert runs/run7_vec_expert/eval/best_model.zip \
                     --init runs/distill/pixel_cnn_v2.pt \
                     --out runs/distill/dagger_r3.pt --rounds 3 --steps 150000
"""
import argparse
import os
import pathlib
import sys

# torch 2.13 optimizer health-check can touch CUDA; hide it when running on CPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from stable_baselines3 import PPO

from env.flappy_env import FlappyEnv
from distill import PixelCnn, DistilledPolicy


def collect(expert, student, n_steps, n_pool=16):
    """parallel-env rollout: student flies, expert labels, k envs per step"""
    envs = [FlappyEnv(obs_mode="dict", shaping=True) for _ in range(n_pool)]
    obs = [e.reset()[0] for e in envs]
    px, ac = [], []
    dev = student.dev
    for _ in range(n_steps // n_pool + 1):
        stack = torch.from_numpy(
            np.stack([o["pixels"] for o in obs]).astype(np.float32) / 255.0
        ).to(dev)
        with torch.no_grad():
            a_s = [int(x) for x in student.net(stack).argmax(-1)]
        a_e, _ = expert.predict(np.stack([o["vec"] for o in obs]),
                                deterministic=True)          # expert labels
        px.extend([o["pixels"].copy() for o in obs])
        ac.extend([int(x) for x in np.atleast_1d(a_e)])
        new_obs = []
        for e, o, a in zip(envs, obs, a_s):
            o2, r, d, t, i = e.step(a)
            if d or t:
                o2, _ = e.reset()
            new_obs.append(o2)
        obs = new_obs
    return (np.array(px, np.uint8), np.array(ac, np.int64))


def train(net, X, Y, dev, epochs=4, bs=256, lr=3e-4):
    n1 = max(float((Y == 1).sum()), 1.0)
    w = torch.tensor([1.0, float((Y != 1).sum()) / n1], device=dev)
    x = torch.from_numpy(X.astype(np.float32) / 255.0).to(dev)
    y = torch.from_numpy(Y).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    n = len(x); m_idx = torch.randperm(n)[: min(n, 30000)]
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = crit(net(x[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        with torch.no_grad():
            acc = (net(x[m_idx]).argmax(-1) == y[m_idx]).float().mean().item()
        print(f"   epoch {ep}: loss {tot / n:.4f} acc {acc:.3f}", flush=True)


def score(net, dev, eps=10):
    env = FlappyEnv(obs_mode="dict", shaping=False)
    sc = []
    for _ in range(eps):
        o, _ = env.reset(); d, st = False, 0
        while not d and st < 30000:
            with torch.no_grad():
                t = torch.from_numpy(o["pixels"].astype(np.float32)[None] / 255.0).to(dev)
                a = int(net(t).argmax(-1))
            o, r, d, tr, info = env.step(a); st += 1
        sc.append(env.state.score)
    return sc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--expert", default="runs/run7_vec_expert/eval/best_model.zip")
    p.add_argument("--init", default="runs/distill/pixel_cnn_v2.pt",
                   help="pretrained pixel student (or 'random')")
    p.add_argument("--out", default="runs/distill/dagger.pt")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--steps", type=int, default=150000, help="rollout steps/round")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    expert = PPO.load(a.expert, device="cpu")
    net = PixelCnn()
    if a.init != "random":
        sd = torch.load(a.init, map_location="cpu")["state"]
        net.load_state_dict(sd)
        print("student init from", a.init)
    net = net.to(dev)
    print(f"expert loaded. rounds={a.rounds} steps/round={a.steps}", flush=True)
    for rd in range(1, a.rounds + 1):
        stud = DistilledPolicy.__new__(DistilledPolicy)
        stud.net, stud.dev = net, dev
        print(f"\n== round {rd}: collecting {a.steps} rollout steps ...", flush=True)
        X, Y = collect(expert, stud, a.steps)
        print(f"   {len(X)} pairs, action=1 ratio {Y.mean():.3f}", flush=True)
        train(net, X, Y, dev)
        sc = score(net, dev)
        print(f"   round {rd} eval scores: {sc} avg {sum(sc)/len(sc):.2f}", flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict()}, a.out)
    print("saved:", a.out)
