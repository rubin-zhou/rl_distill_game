"""Behavior cloning: train a pixel CNN to imitate the expert (vector-trained PPO).

Usage (mlenv), run from repo root:
    python distill.py --expert runs/run7_vec_expert/eval/best_model.zip \
                      --out runs/distill/pixel_cnn.pt --samples 200000
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch
import torch.nn as nn

from env.flappy_env import FlappyEnv


class PixelCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):        # x: (B,H,W,1) float 0..1
        return self.net(x.permute(0, 3, 1, 2))


class DistilledPolicy:
    """predict() compatible with SB3 models, for play.py/eval_scores.py"""

    def __init__(self, ckpt_path, device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                           else "cpu"))
        net = PixelCnn()
        net.load_state_dict(torch.load(ckpt_path, map_location=self.dev)["state"])
        net.eval().to(self.dev)
        self.net = net

    def predict(self, obs, deterministic=True):
        ob = obs["pixels"] if isinstance(obs, dict) else obs
        x = np.asarray(ob, dtype=np.float32)
        if x.ndim == 2:
            x = x[..., None]
        t = torch.from_numpy(x / 255.0).to(self.dev)[None]  # (1,H,W,1)
        with torch.no_grad():
            a = int(self.net(t).argmax(-1).item())
        return np.array([a]), None


def collect(expert_model, n_samples, shaping=True):
    env = FlappyEnv(obs_mode="dict", shaping=shaping)
    o, _ = env.reset()
    px, ac = [], []
    while len(px) < n_samples:
        act, _ = expert_model.predict(o["vec"], deterministic=False)
        px.append(o["pixels"])
        ac.append(int(act))
        o, r, d, t, i = env.step(int(act))
        if d or t:
            o, _ = env.reset()
    return np.array(px, np.uint8), np.array(ac, np.int64)


def distill(expert_path, out_path, samples=200000, device="cpu", epochs=6,
            bs=256, lr=1e-3):
    from stable_baselines3 import PPO
    expert = PPO.load(expert_path)
    print(f"collecting {samples} samples from expert ...", flush=True)
    X, Y = collect(expert, samples, shaping=True)
    print(f"got {X.shape[0]} action=1 ratio {Y.mean():.2f}", flush=True)
    dev = torch.device(device)
    net = PixelCnn().to(dev)
    x = torch.tensor(X.astype(np.float32) / 255.0, device=dev)
    y = torch.tensor(Y, device=dev)
    # train/val split
    n = len(y); idx = torch.randperm(n)
    k = int(0.9 * n)
    train_idx, val_idx = idx[:k], idx[k:]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n1 = max(float((Y == 1).sum()), 1.0)
    n0 = max(float((Y == 0).sum()), 1.0)   # counter the 94% no-flap majority
    w = torch.tensor([1.0, n0 / n1], device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    net.train()
    for ep in range(epochs):
        perm = train_idx[torch.randperm(k)]
        tot = 0.0
        for i in range(0, k, bs):
            b = perm[i:i + bs]
            loss = crit(net(x[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        acc = (net(x[val_idx]).argmax(-1) == y[val_idx]).float().mean().item()
        print(f"epoch {ep}: train loss {tot / k:.4f} val acc {acc:.3f}", flush=True)
    # quick score probe
    env = FlappyEnv(obs_mode="dict", shaping=False)
    net.eval()
    sc = []
    for _ in range(5):
        o, _ = env.reset(); d, stp = False, 0
        while not d and stp < 30000:
            with torch.no_grad():
                inp = torch.tensor(o["pixels"][None].astype(np.float32) / 255.0,
                                   device=dev)
                a = int(net(inp).argmax(-1).item())
            o, r, d, tt, i = env.step(a); stp += 1
        sc.append(env.state.score)
    print("distilled scores (5 eps):", sc, flush=True)
    torch.save({"state": net.state_dict()}, out_path)
    print("saved:", out_path, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--expert", default="runs/run7_vec_expert/eval/best_model.zip")
    p.add_argument("--out", default="runs/distill/pixel_cnn.pt")
    p.add_argument("--samples", type=int, default=200000)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    distill(a.expert, a.out, samples=a.samples, device=device)
