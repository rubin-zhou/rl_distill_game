"""Layer-map based renderer for GameState.

Draws the state into an integer layer array (sky/pipe/ground/bird per pixel),
then maps layers to grayscale (for RL pixels) or RGB (for human display).
Keeps one source of truth for all visuals.
"""
import numpy as np
import cv2

from game.core import (WIDTH, HEIGHT, GROUND_Y, BIRD_X, BIRD_HW, BIRD_HR,
                       PIPE_W, GameState)

SKY, PIPE, GROUND, BIRD = 0, 1, 2, 3

GRAY_VAL = np.array([176, 40, 96, 208], dtype=np.uint8)
RGB_VAL = np.array([(125, 200, 220), (60, 140, 60), (200, 160, 90),
                    (240, 220, 60)], dtype=np.uint8)

_bx = np.arange(WIDTH)[None, :]


def state_layer(state: GameState) -> np.ndarray:
    layer = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    layer[GROUND_Y:, :] = GROUND
    gh2 = state.gap_height / 2.0
    for px, gy, _ in state.pipes:
        x0 = int(max(px, 0))
        x1 = int(min(px + PIPE_W, WIDTH))
        if x1 <= x0:
            continue
        g0 = int(max(gy - gh2, 0))
        g1 = int(min(gy + gh2, HEIGHT))
        layer[:g0, x0:x1] = PIPE
        layer[g1:, x0:x1] = PIPE
    yy = np.arange(HEIGHT)[:, None]
    d2 = (yy - state.bird_y) ** 2 + (_bx - BIRD_X) ** 2
    mask = (d2 <= (BIRD_HR + 2) ** 2) & (_bx > BIRD_X - BIRD_HW - 2)
    layer[mask] = BIRD
    return layer


def state_gray(state: GameState, size=84) -> np.ndarray:
    layer = state_layer(state)
    img = GRAY_VAL[layer]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img[..., None]


def state_rgb(state: GameState, size=None) -> np.ndarray:
    layer = state_layer(state)
    img = RGB_VAL[layer]
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img
