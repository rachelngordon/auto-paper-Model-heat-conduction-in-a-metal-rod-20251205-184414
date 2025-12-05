# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def transient_heat_eq(L=1.0, Nx=101, T_left=100.0, T_initial=0.0, k=1.0, rho=1.0, cp=1.0,
                       t_end=0.5, snapshot_times=None):
    """Simulate 1D transient heat conduction with Dirichlet left BC and insulated right BC.
    Returns times (list) and temperature array T[t_idx, x_idx].
    """
    if snapshot_times is None:
        snapshot_times = [0.0, 0.01, 0.05, 0.2, 0.5]
    dx = L / (Nx - 1)
    alpha = k / (rho * cp)
    dt = 0.4 * dx * dx / alpha  # stability condition for explicit scheme
    Nt = int(np.ceil(t_end / dt)) + 1
    dt = t_end / (Nt - 1)
    # Initialize temperature field
    T = np.full(Nx, T_initial)
    T[0] = T_left
    # Prepare storage for snapshots
    snapshots = {}
    current_time = 0.0
    snapshots[0.0] = T.copy()
    # Time stepping loop
    for n in range(1, Nt):
        T_new = T.copy()
        # interior points
        for i in range(1, Nx - 1):
            T_new[i] = T[i] + alpha * dt / dx**2 * (T[i+1] - 2*T[i] + T[i-1])
        # insulated right boundary (Neumann dT/dx = 0) -> T[N-1] = T[N-2]
        T_new[-1] = T_new[-2]
        # enforce left Dirichlet BC
        T_new[0] = T_left
        T = T_new
        current_time += dt
        # store snapshot if close to requested time
        for ts in snapshot_times:
            if ts not in snapshots and abs(current_time - ts) < dt/2:
                snapshots[ts] = T.copy()
    # Ensure all requested times are present (interpolate if needed)
    for ts in snapshot_times:
        if ts not in snapshots:
            # linear interpolation between nearest stored times
            times = sorted(snapshots.keys())
            lower = max(t for t in times if t < ts)
            upper = min(t for t in times if t > ts)
            w = (ts - lower) / (upper - lower)
            snapshots[ts] = (1-w)*snapshots[lower] + w*snapshots[upper]
    # Return sorted snapshots
    times_sorted = sorted(snapshots.keys())
    T_array = np.vstack([snapshots[t] for t in times_sorted])
    return times_sorted, T_array

def steady_state_temperature(L=1.0, Nx=101, T_left=100.0, T_right=0.0, k=1.0):
    """Solve steady‑state 1D heat equation d/dx(k dT/dx)=0 with Dirichlet BCs.
    Returns temperature array over the rod.
    """
    dx = L / (Nx - 1)
    # Build linear system A T = b for interior nodes
    N_interior = Nx - 2
    A = np.zeros((N_interior, N_interior))
    b = np.zeros(N_interior)
    coeff = k / dx**2
    for i in range(N_interior):
        A[i, i] = -2 * coeff
        if i > 0:
            A[i, i-1] = coeff
        if i < N_interior - 1:
            A[i, i+1] = coeff
    # Apply Dirichlet BCs to RHS
    b[0] -= coeff * T_left
    b[-1] -= coeff * T_right
    # Solve
    T_interior = np.linalg.solve(A, b)
    T = np.empty(Nx)
    T[0] = T_left
    T[1:-1] = T_interior
    T[-1] = T_right
    return T

def experiment_transient():
    times, T_snapshots = transient_heat_eq()
    x = np.linspace(0, 1.0, 101)
    plt.figure(figsize=(8,5))
    for idx, t in enumerate(times):
        plt.plot(x, T_snapshots[idx], label=f"t={t:.2f}s")
    plt.xlabel('Position along rod (m)')
    plt.ylabel('Temperature (°C)')
    plt.title('Transient temperature evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('temp_profile_over_time.png')
    plt.close()

def experiment_steady_state():
    conductivities = [0.5, 1.0, 2.0]
    labels = ['k=0.5 (low)', 'k=1.0 (medium)', 'k=2.0 (high)']
    x = np.linspace(0, 1.0, 101)
    plt.figure(figsize=(8,5))
    for k, lab in zip(conductivities, labels):
        T = steady_state_temperature(k=k)
        plt.plot(x, T, label=lab)
    plt.xlabel('Position along rod (m)')
    plt.ylabel('Temperature (°C)')
    plt.title('Steady‑state temperature distribution for different conductivities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('steady_state_conductivity.png')
    plt.close()
    # Return midpoint temperature for medium conductivity as primary answer
    T_mid = steady_state_temperature(k=1.0)[len(x)//2]
    return T_mid

if __name__ == "__main__":
    experiment_transient()
    answer_value = experiment_steady_state()
    print('Answer:', answer_value)

