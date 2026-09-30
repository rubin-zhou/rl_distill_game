"""Human-play mode:  Space = flap, R = restart, Esc = quit.  Run:
    conda activate mlenv && python game/flappy.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pygame
import cv2
import numpy as np

from env.flappy_env import FlappyEnv
from game.render import state_rgb


def main():
    env = FlappyEnv(obs_mode="pixels", shaping=False)
    pygame.init()
    flags = pygame.RESIZABLE
    fullscreen, stretch = False, False
    pygame.display.set_mode((480, 720), flags=flags)
    win = pygame.display.get_surface()
    pygame.display.set_caption("Flappy Bird (Space=flap, R=restart, "
                               "F=fullscreen, S=stretch, Esc=quit)")
    font = pygame.font.SysFont(None, 40)
    clock = pygame.time.Clock()
    env.reset(seed=123)
    done = False
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN
                                         and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_f, pygame.K_F11):
                fullscreen = not fullscreen
                pygame.display.set_mode(win.get_size(),
                                        pygame.FULLSCREEN if fullscreen
                                        else pygame.RESIZABLE)
            if e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                stretch = not stretch
        flap = pygame.key.get_pressed()[pygame.K_SPACE]
        if done:
            if pygame.key.get_pressed()[pygame.K_r]:
                env.reset(seed=int(clock.get_ticks()) & 0xFFFF)
                done = False
                flap = False
        obs, r, term, trunc, info = env.step(1 if (flap and not done) else 0)
        if term:
            done = True
        w, h = win.get_size()
        if stretch:
            dw, dh, x, y = w, h, 0, 0
        else:
            sc = min(w / 400, h / 600)
            dw = max(1, int(400 * sc))
            dh = max(1, int(600 * sc))
            x, y = (w - dw) // 2, (h - dh) // 2
        base = state_rgb(env.state)
        base = np.transpose(cv2.resize(base, (dw, dh),
                                      interpolation=cv2.INTER_NEAREST),
                            (1, 0, 2))
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(base))
        win.fill((10, 12, 24))
        win.blit(surf, (x, y))
        txt = font.render("score: %d%s" % (
            env.state.score, "  GAME OVER (R to restart)" if done else ""),
            True, (255, 70, 70) if done else (255, 255, 255))
        win.blit(txt, (x + 10, y + 10))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
