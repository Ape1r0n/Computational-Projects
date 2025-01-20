import numpy as np
from dataclasses import dataclass
from pyray import RED, BLUE, draw_circle, init_window, set_target_fps, window_should_close, begin_drawing, clear_background, end_drawing, close_window, WHITE, GREEN
from typing import List, Optional, Tuple


@dataclass
class PhysicsParams:
    phi: float = 0.0005  # Air resistance coefficient
    g: float = 100.0  # Gravity
    dt: float = 0.01  # Time step
    time_steps: int = 200  # Number of simulation steps
    h: float = 0.001  # Step size for shooting method

    def __post_init__(self):
        self.h = min(self.h, 2 / self.phi * 0.95)  # 0.95 makes h slightly smaller than (2m/k)


class Ball:
    def __init__(self, x: float, y: float, vx: float, vy: float, radius: float = 5.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.active = True

    def get_state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy])

    def set_state(self, state: np.ndarray):
        self.x, self.y, self.vx, self.vy = state


class MainGuy:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.params = PhysicsParams()
        self.current_ball: Optional[Ball] = None
        self.targets: List[Tuple[float, float, float]] = []  # x, y, radius
        self.active_targets: List[bool] = []  # Track which targets are still active
        self.shooter_pos = (0, 0)
        self.shooter_radius = 0
        self.current_target_index = 0
        self.shot_fired = False


    def set_shooter(self, pos: Tuple[float, float], radius: float):
        self.shooter_pos = pos
        self.shooter_radius = radius


    def add_target(self, x: float, y: float, radius: float):
        self.targets.append((x, y, radius))
        self.active_targets.append(True)


    def check_collision(self) -> bool:
        if not self.current_ball or not self.current_ball.active:
            return False

        target = self.targets[self.current_ball.target_index]
        if not self.active_targets[self.current_ball.target_index]:
            return False

        dx = self.current_ball.x - target[0]
        dy = self.current_ball.y - target[1]
        distance = np.sqrt(dx * dx + dy * dy)

        return distance < (self.current_ball.radius + target[2])

    def shoot_next_ball(self) -> bool:
        active_targets = [(i, t) for i, t in enumerate(self.targets) if self.active_targets[i]]
        if not active_targets:
            return False

        active_targets.sort(key=lambda t: np.hypot(t[1][0] - self.shooter_pos[0], t[1][1] - self.shooter_pos[1]))
        closest_index, target = active_targets[0]
        vx, vy = shooting_method(self.shooter_pos[0], self.shooter_pos[1], target[0], target[1], self.params)

        self.current_ball = Ball(self.shooter_pos[0], self.shooter_pos[1], vx, vy)
        self.current_ball.target_index = closest_index
        self.shot_fired = True
        return True


    def simulate(self):
        init_window(self.width, self.height, "THE MainGuy Simulation")
        set_target_fps(60)

        print("Simulation started")
        cnt = 1
        while not window_should_close():
            begin_drawing()
            clear_background(WHITE)

            for i, (tx, ty, tr) in enumerate(self.targets):
                if self.active_targets[i]:
                    draw_circle(int(tx), int(ty), tr, RED)

            draw_circle(int(self.shooter_pos[0]), int(self.shooter_pos[1]), self.shooter_radius, GREEN)

            if not self.shot_fired:
                if not self.shoot_next_ball():
                    if self.current_target_index >= len(self.targets):
                        break
            elif self.current_ball and self.current_ball.active:
                rk4_step(self.current_ball, self.params)
                draw_circle(int(self.current_ball.x), int(self.current_ball.y), self.current_ball.radius, BLUE)

                if self.check_collision():
                    self.active_targets[self.current_ball.target_index] = False
                    self.current_ball.active = False
                    self.shot_fired = False
                    self.current_target_index += 1

                if (self.current_ball.x < 0 or self.current_ball.x > self.width or
                        self.current_ball.y < 0 or self.current_ball.y > self.height):
                    self.current_ball.active = False
                    self.shot_fired = False

            print(f"Frame {cnt} finished")
            cnt += 1
            end_drawing()

        close_window()



def acceleration(v: Tuple[float, float], params: PhysicsParams) -> Tuple[float, float]:
    vx, vy = v
    mag = np.sqrt(vx * vx + vy * vy)
    return (-params.phi * vx * mag, params.g - params.phi * vy * mag)


def derivative_vector(state: np.ndarray, params: PhysicsParams) -> np.ndarray:
    x, y, vx, vy = state
    ax, ay = acceleration((vx, vy), params)
    return np.array([vx, vy, ax, ay])


def rk4_step(ball: Ball, params: PhysicsParams) -> None:
    state = ball.get_state()
    k1 = derivative_vector(state, params)
    k2 = derivative_vector(state + 0.5 * params.dt * k1, params)
    k3 = derivative_vector(state + 0.5 * params.dt * k2, params)
    k4 = derivative_vector(state + params.dt * k3, params)

    new_state = state + (params.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    ball.set_state(new_state)


def heun_rk2_step(ball: Ball, params: PhysicsParams) -> None:
    state = ball.get_state()

    # Predictor step (Euler's method)
    k1 = derivative_vector(state, params)
    predictor = state + params.dt * k1

    # Corrector step (average slope)
    k2 = derivative_vector(predictor, params)

    # Update state
    new_state = state + (params.dt / 2.0) * (k1 + k2)
    ball.set_state(new_state)


def shooting_method(x0: float, y0: float, targetx: float, targety: float, params: PhysicsParams) -> Tuple[float, float]:
    vx0, vy0 = 0.0, 0.0

    for _ in range(5):  # Maximum iterations for shooting method
        b = Ball(x0, y0, vx0, vy0)
        b1 = Ball(x0, y0, vx0 + params.h, vy0)
        b2 = Ball(x0, y0, vx0, vy0 + params.h)

        # Simulate trajectories
        for _ in range(params.time_steps):
            rk4_step(b, params)
            rk4_step(b1, params)
            rk4_step(b2, params)

        # Calculate the Jacobian and errors
        e1, e2 = targetx - b.x, targety - b.y
        j11 = (b1.x - b.x) / params.h
        j12 = (b2.x - b.x) / params.h
        j21 = (b1.y - b.y) / params.h
        j22 = (b2.y - b.y) / params.h
        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-5:
            break

        # Update velocities
        dvx = (j22 * e1 - j12 * e2) / det
        dvy = (-j21 * e1 + j11 * e2) / det
        vx0 += dvx
        vy0 += dvy

    return vx0, vy0