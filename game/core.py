"""Pure-logic Flappy Bird core. No rendering, no pygame -- frame-based physics."""
import random

WIDTH = 400
HEIGHT = 600
GROUND_Y = HEIGHT - 90
CEIL_Y = 8
BIRD_X = 96
BIRD_HW = 15  # collision half-width
BIRD_HR = 11  # collision half-height
GRAVITY = 0.45
MAX_FALL = 11.0
FLAP_V = -6.8
PIPE_W = 52
PIPE_MARGIN = 80


class GameState:
    """Deterministic game state stepped frame-by-frame.

    step(action) advances one frame (1/60 s at "real time", used as the RL dt).
    """

    def __init__(self, seed=None, gap_height=110, speed=3.0, pipe_interval=204.0):
        self.rng = random.Random(seed)
        self.gap_height = float(gap_height)
        self.speed = float(speed)
        self.pipe_interval = float(pipe_interval)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.bird_y = HEIGHT * 0.45
        self.vy = 0.0
        self.score = 0
        self.frame = 0
        self.alive = True
        self.pipes = []
        last_x = BIRD_X + 200
        while last_x < WIDTH + self.pipe_interval:
            if not self.pipes or last_x <= self.pipes[-1][0]:
                last_x = (WIDTH if not self.pipes
                          else self.pipes[-1][0] + self.pipe_interval)
            self.pipes.append([last_x, self._rand_gap_y(), False])
            last_x += self.pipe_interval
        return self

    def _rand_gap_y(self):
        gh = self.gap_height
        lo = gh / 2 + PIPE_MARGIN
        hi = GROUND_Y - gh / 2 - PIPE_MARGIN
        return self.rng.uniform(lo, hi)

    def step(self, action):
        if not self.alive:
            return
        self.frame += 1

        self.vy = min(self.vy + GRAVITY, MAX_FALL)
        if action == 1:
            self.vy = FLAP_V
        self.bird_y += self.vy

        if self.bird_y < CEIL_Y:
            self.bird_y = CEIL_Y
            self.vy = max(self.vy, 0.0)

        for p in self.pipes:
            p[0] -= self.speed
        if not self.pipes:
            self.pipes.append([WIDTH, self._rand_gap_y(), False])
        elif self.pipes[-1][0] < WIDTH - self.pipe_interval:
            self.pipes.append([WIDTH, self._rand_gap_y(), False])

        bx = BIRD_X
        self.pipes = [p for p in self.pipes if p[0] + PIPE_W > -20]
        for p in self.pipes:
            p_x, gap_y, passed = p
            if (not p[2] and p_x + PIPE_W < bx - BIRD_HW):
                p[2] = True
                self.score += 1
            overlap_x = p_x < bx + BIRD_HW and p_x + PIPE_W > bx - BIRD_HW
            if overlap_x:
                gh2 = self.gap_height / 2.0
                if self.bird_y - BIRD_HR < gap_y - gh2 or self.bird_y + BIRD_HR > gap_y + gh2:
                    self.alive = False
                    break
        if self.alive and self.bird_y + BIRD_HR >= GROUND_Y:
            self.alive = False
        return self

    @property
    def done(self):
        return not self.alive

    def next_pipes(self, n=2):
        """Forthcoming pipes sorted by x, each (dx, gap_dy) relative to bird."""
        out = []
        for p in sorted(self.pipes):
            dx = p[0] - BIRD_X
            if dx > 2 * self.pipe_interval:
                continue
            out.append((dx, p[1] - self.bird_y))
            if len(out) >= n:
                break
        if not out:
            out = [(self.pipe_interval, 0.0)]
        return out
