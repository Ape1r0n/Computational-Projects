import cv2
import numpy as np
import sys


def detect_ball_and_velocity(frame1: np.ndarray, frame2: np.ndarray, p2m: float = 1.0, f_time: float = 1 / 30.0):
    gray_scaled_1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_scaled_2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    blurred_1 = cv2.GaussianBlur(gray_scaled_1, (3, 3), 0)
    blurred_2 = cv2.GaussianBlur(gray_scaled_2, (3, 3), 0)

    edges_1 = cv2.Canny(blurred_1, 30, 100)
    edges_2 = cv2.Canny(blurred_2, 30, 100)
    contours_1, hierarchy_1 = cv2.findContours(edges_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_2, hierarchy_2 = cv2.findContours(edges_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def find_ball_position(contours):
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if 0.5 < circularity < 1.5:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return cx, cy
        return None

    pos_1 = find_ball_position(contours_1)
    pos_2 = find_ball_position(contours_2)


    if pos_1 is None or pos_2 is None:
        return None, (np.inf, np.inf)

    v_x = (pos_2[0] - pos_1[0]) * p2m / f_time
    v_y = (pos_2[1] - pos_1[1]) * p2m / f_time

    return pos_2, [v_x, v_y]


def central_FD(v_prev: float, v_next: float, dt: float) -> float:
    return (v_next - v_prev) / (2 * dt)


def get_k_over_m_and_g(v_x, v_y, dv_x, dv_y):
    return (-1 / (v_x * np.sqrt(v_x ** 2 + v_y ** 2)) * dv_x,
            v_y / (v_x if v_x != 0 else 1e-10) * dv_x - dv_y)


def interpolate_missing_values(values):
    values = np.array(values, dtype=float)
    nan_indices = np.where(np.isnan(values))[0]
    valid_indices = np.where(~np.isnan(values))[0]

    if len(valid_indices) > 1:
        values[nan_indices] = np.interp(nan_indices, valid_indices, values[valid_indices])
    return values


def visualize_detection(frame, ball_pos, velocity, k_over_m=None, g=None):
    output = frame.copy()
    if ball_pos is None or np.isinf(ball_pos[0]) or np.isinf(ball_pos[1]):
        return output

    ball_pos = tuple(map(int, ball_pos))
    cv2.circle(output, ball_pos, 10, (0, 255, 0), 2)
    scale = 20
    end_point = (int(ball_pos[0] + velocity[0] * scale), int(ball_pos[1] + velocity[1] * scale))
    cv2.arrowedLine(output, ball_pos, end_point, (0, 0, 255), 2)

    speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    cv2.putText(output, f'Speed: {speed:.2f} m/s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if k_over_m is not None:
        cv2.putText(output, f'k/m: {k_over_m:.4f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if g is not None:
        cv2.putText(output, f'g: {g:.4f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return output



if __name__ == "__main__":
    PATH = 'Not Free Fall.mp4'
    cap = cv2.VideoCapture(PATH)

    if not cap.isOpened():
        print("Error: Could not open video file")
        sys.exit(1)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        sys.exit(1)

    ret, current_frame = cap.read()
    if not ret:
        print("Error: Could not read second frame")
        cap.release()
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    temp_dv_x = []
    temp_dv_y = []
    temp_k_over_m = []
    temp_g = []
    temp_v_x = []
    temp_v_y = []
    temp_ball_pos = []

    _, prev_velocity = detect_ball_and_velocity(prev_frame, current_frame)
    prev_frame = current_frame.copy()

    current_k_over_m = None
    current_g = None

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        ball_pos, current_velocity = detect_ball_and_velocity(prev_frame, current_frame)

        if ball_pos is not None and not np.isinf(current_velocity[0]) and not np.isinf(prev_velocity[0]) and current_velocity[0] != 0:
            temp_ball_pos.append(ball_pos)
            temp_v_x.append(current_velocity[0])
            temp_v_y.append(current_velocity[1])

            d_x = central_FD(prev_velocity[0], current_velocity[0], 1)
            d_y = central_FD(prev_velocity[1], current_velocity[1], 1)
            temp_dv_x.append(d_x)
            temp_dv_y.append(d_y)

            current_k_over_m, current_g = get_k_over_m_and_g(current_velocity[0], current_velocity[1], d_x, d_y)
            temp_k_over_m.append(current_k_over_m)
            temp_g.append(current_g)
        else:
            current_k_over_m = None
            current_g = None

        output_frame = visualize_detection(current_frame, ball_pos, current_velocity, current_k_over_m, current_g)
        cv2.imshow('Ball Tracking', output_frame)
        out.write(output_frame)

        prev_frame = current_frame.copy()
        prev_velocity = current_velocity

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    dv_x = np.array(temp_dv_x)
    dv_y = np.array(temp_dv_y)
    k_over_m = np.array(temp_k_over_m)
    g = np.array(temp_g)
    v_x = np.array(temp_v_x)
    v_y = np.array(temp_v_y)
    ball_pos = np.array(temp_ball_pos)

    print(f"Number of valid frames: {len(dv_x)}")
    print(f"Average v_x: {np.mean(v_x):.4f}")
    print(f"Average v_y: {np.mean(v_y):.4f}")
    print(f"Average dv_x: {np.mean(dv_x):.4f}")
    print(f"Average dv_y: {np.mean(dv_y):.4f}")
    print(f"Average k/m: {np.mean(k_over_m):.4f}")
    print(f"Average g: {np.mean(g):.4f}")

    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(12, 6))
    # plt.scatter([x[0] for x in ball_pos], [x[1] for x in ball_pos], c='red', marker='x')
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Ball Positions")
    # plt.show()
