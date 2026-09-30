"""PPO.load wrapper that tolerates checkpoints saved with our 2-group warmup
optimizer (group0=policy frozen, group1=value). Plain PPO.load rebuilds a
single-group optimizer and dies on state_dict mismatch; for eval/resume we
don't need the optimizer state anyway, so strip it and retry.
"""
import tempfile
import zipfile

from stable_baselines3 import PPO


def load_ppo(path, **kwargs):
    try:
        return PPO.load(path, **kwargs)
    except ValueError:
        with zipfile.ZipFile(path) as zin, \
                zipfile.ZipFile(path) as zin2, \
                tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
            with zipfile.ZipFile(tf.name, "w", zipfile.ZIP_DEFLATED) as zout:
                for it in zin.infolist():
                    if it.filename.endswith("optimizer.pth"):
                        continue
                    zout.writestr(it, zin.read(it.filename))
        return PPO.load(tf.name, **kwargs)
