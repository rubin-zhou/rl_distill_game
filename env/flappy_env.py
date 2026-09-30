"""Gymnasium wrapper around the Flappy Bird core.

obs_mode:
  - "pixels": Box(84, 84, 1) uint8 grayscale (default, used for training)
  - "vector": Box(7,) float32, normalized state vector (fast, for experiments)

Reward: +1 alive per step, +10 pipe passed, -10 crash.
Optional shaping (gamma-correct potential based) helps early exploration.
"""
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import gymnasium
from gymnasium import spaces

from game.core import (WIDTH, HEIGHT, GROUND_Y, BIRD_X, BIRD_HR,
                       CEIL_Y, MAX_FALL, GameState)
from game.render import state_gray

GAMMA = 0.99
PASS_REWARD = 20.0
CRASH_REWARD = -20.0
ALIVE_REWARD = 0.5
SHAPE_ALPHA = 1.0
SHAPE_RANGE = 180.0  # px: shaping active only when the next pipe is this close


def _vec_obs(state):
    pipes = state.next_pipes(2)
    while len(pipes) < 2:
        pipes.append((WIDTH, 0.0))
    (dx1, dy1), (dx2, dy2) = pipes[0], pipes[1]
    sgn1 = 1.0 if state.bird_y + BIRD_HR >= GROUND_Y - 1e-9 else 0.0
    return np.array([
        (state.bird_y - HEIGHT / 2) / (HEIGHT / 2),
        state.vy / MAX_FALL,
        min(dx1, 2 * WIDTH) / WIDTH,
        dy1 / (HEIGHT / 2),
        min(dx2, 2 * WIDTH) / WIDTH,
        dy2 / (HEIGHT / 2),
        sgn1,
    ], dtype=np.float32)


class FlappyEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, obs_mode="pixels", gap_height=110, speed=3.0,
                 pipe_interval=204.0, shaping=True, frame_skip=1,
                 display_scale=1.2):
        super().__init__()
        self.obs_mode = obs_mode
        self.shaping = shaping
        self.frame_skip = int(frame_skip)
        self.gap_height = gap_height
        self.state = GameState(seed=None, gap_height=gap_height, speed=speed,
                               pipe_interval=pipe_interval)
        self._state_kwargs = dict(gap_height=gap_height, speed=speed,
                                  pipe_interval=pipe_interval)
        if obs_mode == "pixels":
            self.observation_space = spaces.Box(0, 255, (84, 84, 1), np.uint8)
        elif obs_mode == "vector":
            self.observation_space = spaces.Box(-1, 1, (7,), np.float32)
        elif obs_mode == "dict":
            # pixels + vector sidecar: CNN gets a gradient it can actually use
            self.observation_space = spaces.Dict({
                "pixels": spaces.Box(0, 255, (84, 84, 1), np.uint8),
                "vec": spaces.Box(-1, 1, (7,), np.float32),
            })
        else:
            raise ValueError(obs_mode)
        self.action_space = spaces.Discrete(2)
        self._prev_pot = 0.0
        self.render_mode = "rgb_array"
        self._win = None
        self._surface = None
        self._font = None
        self._display_scale = display_scale
        self._fullscreen = False
        self._stretch = False       # F fills window, S stretches w/o letterbox
        self.quit_requested = False

    def _obs(self):
        if self.obs_mode == "pixels":
            return state_gray(self.state)
        if self.obs_mode == "dict":
            return {"pixels": state_gray(self.state), "vec": _vec_obs(self.state)}
        return _vec_obs(self.state)

    def _potential(self):
        if not self.shaping:
            return 0.0
        dx, dy = self.state.next_pipes(1)[0]
        if dx > SHAPE_RANGE:  # far away: no reward for hovering, stay neutral
            return 0.0
        weight = (SHAPE_RANGE - dx) / SHAPE_RANGE  # state-dependent gate,
        return -abs(dy) / (self.gap_height / 2.0) * SHAPE_ALPHA * weight  # still potential-based

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = GameState(seed=seed if seed is not None else int(self.np_random.integers(0, 2**31)),
                               **self._state_kwargs)
        self._prev_pot = self._potential()
        return self._obs(), {"score": 0}

    def step(self, action):
        total_r = 0.0
        for _ in range(self.frame_skip):
            before_score = self.state.score
            self.state.step(int(action))
            alive_r = ALIVE_REWARD if self.state.alive else 0.0
            pass_r = PASS_REWARD * (self.state.score - before_score)
            pot = self._potential()
            shape_r = (pot - GAMMA * self._prev_pot)
            self._prev_pot = pot
            r = alive_r + pass_r + shape_r
            if not self.state.alive:
                r = CRASH_REWARD
            total_r += r
            if self.state.done:
                break
        info = {"score": self.state.score}
        return self._obs(), float(total_r), bool(self.state.done), False, info

    def render(self, mode="rgb_array"):
        from game.render import state_rgb
        base = state_rgb(self.state)
        if mode == "human":
            import pygame
            if self._win is None:
                pygame.init()
                size = (int(WIDTH * self._display_scale),
                        int(HEIGHT * self._display_scale))
                pygame.display.set_mode(size, flags=pygame.RESIZABLE)
                self._win = pygame.display.get_surface()
                pygame.display.set_caption(
                    "Flappy Bird - RL   (F=fullscreen, S=stretch, "
                    "Esc=quit, drag edges to resize)")
                self._font = pygame.font.SysFont(None, 30)
            # scale the scene to the ACTUAL window surface size every frame:
            # WM/HiDPI scaling can make the buffer bigger than what we asked
            # for, and a fixed-size blit then leaves unpainted black bands.
            w, h = self._win.get_size()
            if self._stretch:
                dw, dh, x, y = w, h, 0, 0
            else:
                sc = min(w / WIDTH, h / HEIGHT)
                dw = max(1, int(WIDTH * sc))
                dh = max(1, int(HEIGHT * sc))
                x, y = (w - dw) // 2, (h - dh) // 2
            img = np.transpose(cv2.resize(base, (dw, dh),
                                          interpolation=cv2.INTER_NEAREST),
                               (1, 0, 2))
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(img))
            self._win.fill((10, 12, 24))
            self._win.blit(surf, (x, y))
            self._win.blit(self._font.render(f"score: {self.state.score}",
                                             True, (255, 255, 255)),
                           (x + 8, y + 8))
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (
                        e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    self.quit_requested = True
                elif e.type == pygame.KEYDOWN and e.key in (pygame.K_f,
                                                            pygame.K_F11):
                    self._fullscreen = not self._fullscreen
                    pygame.display.set_mode(self._win.get_size(),
                                            pygame.FULLSCREEN
                                            if self._fullscreen
                                            else pygame.RESIZABLE)
                    self._win = pygame.display.get_surface()
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                    self._stretch = not self._stretch
            pygame.display.flip()
            pygame.event.pump()
        return base
