import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def gm_model(y, t, p1, p2, p3, p4, Gb, Ib, D_t, u_t):
    G, X, I = y
    dGdt = -(p1 + X) * G + p1 * Gb + D_t(t)
    dXdt = -p2 * X + p3 * (I - Ib)
    dIdt = -p4 * (I - Ib) + u_t(t)
    return [dGdt, dXdt, dIdt]

def simulate_glucose_insulin(p, Gb=90, Ib=15, T=300, dt=1.0):
    p1, p2, p3, p4 = p
    times = np.arange(0, T + dt, dt)

    D_t = lambda t: 20.0 if 50 <= t <= 60 else 0.0
    u_t = lambda t: 1.0 if 100 <= t <= 110 else 0.0

    y0 = [Gb, 0.0, Ib]
    solution = odeint(gm_model, y0, times, args=(p1, p2, p3, p4, Gb, Ib, D_t, u_t))
    G, X, I = solution.T

    return times, G, X, I, D_t, u_t

def ekf_step(x, P, z, u, D, dt, params, R, Q):
    p1, p2, p3, p4, Gb, Ib = params

    G, X, I = x
    dG = -(p1 + X) * G + p1 * Gb + D
    dX = -p2 * X + p3 * (I - Ib)
    dI = -p4 * (I - Ib) + u
    x_pred = x + dt * np.array([dG, dX, dI])

    A = np.array([
        [-(p1 + X), -G, 0],
        [0, -p2, p3],
        [0, 0, -p4]
    ])
    A = np.eye(3) + dt * A

    P_pred = A @ P @ A.T + Q
    H = np.eye(3)
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x_upd = x_pred + K @ (z - x_pred)
    P_upd = (np.eye(3) - K @ H) @ P_pred

    return x_upd, P_upd
    
# Mutual Information (MI) approximation
def mi_approx(P):
    return 0.5 * np.log(np.linalg.det(P) + 1e-6)

# Gradient Descent Parameter Learning
def update_parameters(params, x, z, u, D, Gb, Ib, dt, lr):
    params = np.array(params)  # Ensure NumPy array
    p1, p2, p3, p4 = params
    G, X, I = x
    pred = np.array([
        -(p1 + X) * G + p1 * Gb + D,
        -p2 * X + p3 * (I - Ib),
        -p4 * (I - Ib) + u
    ])
    error = pred * dt - (z - x)
    grad = np.array([
        (-G + Gb) * dt * error[0],
        -X * dt * error[1],
        (I - Ib) * dt * error[1],
        -(I - Ib) * dt * error[2]
    ])
    return params - lr * grad


def select_active_input(x, P, u_candidates, D, dt, theta, R, Q):
    best_u, max_mi = None, -np.inf
    for u in u_candidates:
        x_pred, P_pred = ekf_step(x.copy(), P.copy(), x, u, D, dt, theta, R, Q)
        mi = 0.5 * np.log(np.linalg.det(P_pred) + 1e-6)
        if mi > max_mi:
            best_u, max_mi = u, mi
    return best_u

if __name__ == "__main__":
    true_params = [0.01, 0.02, 0.0005, 0.1]
    est_params = [0.02, 0.01, 0.0003, 0.2]  # initial guess
    Gb, Ib = 90, 15
    T, dt = 300, 1.0
    times, G_true, X_true, I_true, D_t, u_t = simulate_glucose_insulin(true_params, Gb, Ib, T, dt)

    x_est = np.array([Gb, 0.0, Ib])
    P = np.eye(3)
    Q = 1e-3 * np.eye(3)
    R = 1e-1 * np.eye(3)
    x_hist, P_hist, theta_hist = [], [], []

    for i, t in enumerate(times):
        D = D_t(t)
        u_candidates = np.linspace(0.5, 1.5, 5)
        
        theta = np.concatenate([est_params, [Gb, Ib]])
        u = select_active_input(x_est, P, u_candidates, D, dt, theta, R, Q)

        z = np.array([G_true[i], X_true[i], I_true[i]]) + np.random.multivariate_normal(np.zeros(3), R)

        x_est, P = ekf_step(x_est, P, z, u, D, dt, theta, R, Q)
        est_params = update_parameters(est_params, x_est, z, u, D, Gb, Ib, dt, 1e-2)

        x_hist.append(x_est.copy())
        P_hist.append(P.copy())
        theta_hist.append(est_params.copy())

    x_hist = np.array(x_hist)
    theta_hist = np.array(theta_hist)

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(times, G_true, 'b-', label='True G')
    plt.plot(times, x_hist[:, 0], 'r--', label='EKF G')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times, X_true, 'b-', label='True X')
    plt.plot(times, x_hist[:, 1], 'r--', label='EKF X')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times, I_true, 'b-', label='True I')
    plt.plot(times, x_hist[:, 2], 'r--', label='EKF I')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for j in range(4):
        plt.plot(times, theta_hist[:, j], label=f'p{j+1}')
    plt.axhline(true_params[0], linestyle='--', color='gray')
    plt.title('Parameter Estimates Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()
