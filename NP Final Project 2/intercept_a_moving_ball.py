from typing import cast
import numpy as np
import cv2
from numpy._core.multiarray import ndarray
from sklearn.cluster import DBSCAN
from pyray import *
import copy

H: float = 0.001  # constant. small value for numerical differentiation


class Vector:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float):
        return Vector(self.x * other, self.y * other)

    def __rmul__(self, other: float):
        return self * other

    def __truediv__(self, other: float):
        return Vector(self.x / other, self.y / other)


class Ball:
    def __init__(self, p: Vector, v: Vector, phi, g):
        self.p, self.v = copy.deepcopy(p), copy.deepcopy(v)
        self.phi, self.g = phi, g


def read_data_points(file_path: str) -> tuple[list[Vector], float, tuple[int, int]]:
    cap = cv2.VideoCapture(file_path)
    dims = None
    count = 0
    acc = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        dims = frame.shape
        frame = np.array(frame).astype(np.float32)
        if acc is None:
            acc = frame
        else:
            x = [acc, frame]
            acc = np.sum(np.array(x), axis=0)
        count += 1

    cap.release()

    bg = cast(ndarray, acc) / count
    cap = cv2.VideoCapture(file_path)
    count = 0
    all_centers = []
    radius_acc = 0
    radius_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = np.array(frame).astype(np.float32)
        diff = cv2.absdiff(frame, bg)
        bw_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        projectile_points = np.column_stack(np.where(bw_diff > 20)[::-1])
        scanner = DBSCAN(eps=5.0)
        if len(projectile_points) == 0:
            continue
        labels = scanner.fit_predict(projectile_points)
        for label in set(labels):
            if label == -1:
                continue
            cluster = projectile_points[labels == label]
            center = np.mean(cluster, axis=0)
            radius_acc += np.sqrt(((cluster - center) ** 2).sum(axis=1).max())
            radius_count += 1
            all_centers.append(Vector(center[0], center[1]))

        count += 1

    return all_centers, radius_acc / radius_count, cast(tuple[int, int], dims)


def acceleration(v: Vector, phi: float, g: float) -> Vector:
    mag = (v.x * v.x + v.y * v.y) ** 0.5
    return Vector(-v.x * phi * mag, g - v.y * phi * mag)


def rk4_step(b: Ball, dt: float):
    k1_p = b.v
    k1_v = acceleration(b.v, b.phi, b.g)
    k2_p = b.v + k1_v * (dt / 2)
    k2_v = acceleration(b.v + k1_v * (dt / 2), b.phi, b.g)
    k3_p = b.v + k2_v * (dt / 2)
    k3_v = acceleration(b.v + k2_v * (dt / 2), b.phi, b.g)
    k4_p = b.v + k3_v * dt
    k4_v = acceleration(b.v + k3_v * dt, b.phi, b.g)
    b.p += (k1_p + 2 * k2_p + 2 * k3_p + k4_p) * (dt / 6)
    b.v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * (dt / 6)


def heun_rk2_step(b: Ball, dt: float):
    k1_p = b.v
    k1_v = acceleration(b.v, b.phi, b.g)
    k2_p = b.v + k1_v * dt
    k2_v = acceleration(b.v + k1_v * dt, b.phi, b.g)
    b.p += (k1_p + k2_p) * (dt / 2)
    b.v += (k1_v + k2_v) * (dt / 2)



def shooting_method(b0: Ball, target: Vector, dt: float, steps: int) -> Vector:
    v = Vector(b0.v.x, b0.v.y)

    for sakura in range(10):
        b = Ball(b0.p, v, b0.phi, b0.g)
        bx = Ball(b0.p, v + Vector(H, 0), b0.phi, b0.g)
        by = Ball(b0.p, v + Vector(0, H), b0.phi, b0.g)

        for useless2 in range(steps):
            rk4_step(b, dt)
            rk4_step(bx, dt)
            rk4_step(by, dt)

        error = target - b.p
        j_1 = (bx.p - b.p) / H
        j_2 = (by.p - b.p) / H
        det = j_1.x * j_2.y - j_2.x * j_1.y
        if abs(det) < 1e-4:
            break
        v += Vector(
            (j_2.y * error.x - j_2.x * error.y) / det,
            (-j_1.y * error.x + j_1.x * error.y) / det
        )

    return v


def get_g_and_phi(p: list[Vector]) -> tuple[float, float]:
    n = len(p)

    v = [Vector(0, 0) for _ in range(n)]
    for i in range(1, n - 1):
        v[i] = (p[i + 1] - p[i - 1]) / 2.0
    v[0] = v[1]
    v[n - 1] = v[n - 2]

    a = [Vector(0, 0) for _ in range(n)]
    for i in range(1, n - 1):
        a[i] = (v[i + 1] - v[i - 1]) / 2.0
    a[0] = a[1]
    a[n - 1] = a[n - 2]

    avg_v = v[1] + v[n - 2]
    avg_a = Vector(0, 0)
    for i in range(2, n - 2):
        avg_v += v[i]
        avg_a += a[i]
    avg_v /= (n - 2)
    avg_a /= (n - 4)

    phi = max((-1.0 / (avg_v.x * (avg_v.x ** 2 + avg_v.y ** 2) ** 0.5) * avg_a.x), 0.0)
    g = avg_a.x * avg_v.y / avg_v.x + avg_a.y

    return g, phi


def get_initial_ball(p: list[Vector]) -> Ball:
    position = p[0]
    g, phi = get_g_and_phi(p)
    velocity = shooting_method(Ball(position, Vector(0, 0), phi, g), p[-1], 1, len(p) - 1)
    return Ball(position, velocity, phi, g)


if __name__ == "__main__":
    points, radius, dims = read_data_points("report/video 1.mp4")
    scale = min(1.0 / dims[0], 1.0 / dims[1])
    points = [point * scale for point in points]

    ball = get_initial_ball(points)
    ball_copy = copy.deepcopy(ball)
    for i in range(200):
        rk4_step(ball_copy, 1)

    bullet_v = shooting_method(Ball(Vector(0.5, 1.0), Vector(0, 0), ball.phi, ball.g), ball_copy.p, 1, 200)
    bullet = Ball(Vector(0.5, 1.0), bullet_v, ball.phi, ball.g)

    init_window(640, 480, "Intercept A Moving Ball")
    set_target_fps(100)

    while not window_should_close():
        begin_drawing()
        clear_background(WHITE)
        draw_circle(int(ball.p.x * 480), int(ball.p.y * 480), radius, RED)
        draw_circle(int(bullet.p.x * 480), int(bullet.p.y * 480), 10, PURPLE)

        for p in points:
            draw_circle(int(p.x * 480), int(p.y * 480), 2, BLUE)
        rk4_step(bullet, 1)
        rk4_step(ball, 1)

        dist = np.sqrt((bullet.p.x - ball.p.x) ** 2 + (bullet.p.y - ball.p.y) ** 2)
        if dist < 1e-5:
            print("<----- Missile hit the target! ----->")
            break

        end_drawing()
    close_window()