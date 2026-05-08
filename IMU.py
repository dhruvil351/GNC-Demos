import numpy as np
import matplotlib.pyplot as plt

def solve_strapdown_stable():
    # --- Constants ---
    Re = 6378137.0          # Earth Radius (m)
    We = 7.292115e-5        # Earth Rotation Rate (rad/s)
    mu = 3.986004418e14     # Standard Gravitational Parameter
    
    # --- Simulation Setup ---
    duration = 24 * 3600    # 1 Day (Full Circle)
    dt = 1.0                # 1.0s Time Step
    
    # --- Sensor Inputs (Constant in Body Frame) ---
    # The IMU measures the reaction force holding it stationary against gravity
    # and centripetal acceleration.
    g_mag = mu / (Re**2)
    ac_mag = (We**2) * Re
    f_mag = g_mag - ac_mag
    
    # Body Frame: X-axis points UP (Outward), Z-axis is rotation axis
    f_body = np.array([f_mag, 0.0, 0.0]) 
    w_body = np.array([0.0, 0.0, We])
    
    # --- Initial State ---
    r0 = np.array([Re, 0.0, 0.0])
    v0 = np.array([0.0, Re*We, 0.0]) # Tangential velocity
    C0 = np.eye(3).flatten()         # Flattened Identity Matrix
    state = np.concatenate([r0, v0, C0])
    
    # --- Derivatives Function (RK4) ---
    def derivatives(t, y):
        r = y[0:3]
        v = y[3:6]
        C = y[6:15].reshape((3,3))
        
        # 1. Kinematics
        dr = v
        
        # 2. Attitude Dynamics
        wx, wy, wz = w_body
        Omega = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
        dC = C @ Omega
        
        # 3. Navigation Dynamics
        f_inertial = C @ f_body
        r_norm = np.linalg.norm(r)
        g_inertial = - (mu / r_norm**3) * r
        
        # --- VERTICAL DAMPING (The Fix) ---
        # Real INS systems use barometric aiding to stabilize the vertical channel.
        # Without this, d_err_double_dot = 2g/R * err (Exponential Instability).
        # We apply a PD controller to the radial acceleration to stabilize altitude.
        k_p = 1e-6 # Position gain
        k_d = 1.0  # Velocity gain
        
        radial_unit = r / r_norm
        radial_vel = np.dot(v, radial_unit)
        radial_error = r_norm - Re
        
        damping_acc = - (k_p * radial_error + k_d * radial_vel) * radial_unit
        
        dv = f_inertial + g_inertial + damping_acc
        
        return np.concatenate([dr, dv, dC.flatten()])

    # --- RK4 Integration Loop ---
    t_eval = np.arange(0, duration, dt)
    results = []
    
    curr_state = state.copy()
    
    print("Running 24-hour Strapdown Simulation (With Vertical Damping)...")
    
    for t in t_eval:
        # Save results every 60 steps (1 minute) to reduce memory
        if t % 60 == 0:
            results.append(curr_state[0:3])
            
        k1 = derivatives(t, curr_state)
        k2 = derivatives(t + dt/2, curr_state + dt/2 * k1)
        k3 = derivatives(t + dt/2, curr_state + dt/2 * k2)
        k4 = derivatives(t + dt, curr_state + dt * k3)
        
        curr_state = curr_state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Re-Orthonormalize DCM
        C_temp = curr_state[6:15].reshape((3,3))
        u, s, vh = np.linalg.svd(C_temp)
        C_clean = u @ vh
        curr_state[6:15] = C_clean.flatten()

    # --- Plotting ---
    traj = np.array(results)
    final_r = np.linalg.norm(traj[-1])
    drift = final_r - Re
    
    print(f"Simulation Complete.")
    print(f"Final Radius: {final_r:.4f} m")
    print(f"Radial Drift: {drift:.4f} m")
    
    plt.figure(figsize=(6, 6))
    plt.plot(traj[:,0], traj[:,1], label='Integrated Path')
    
    # Earth Circle for reference
    angle = np.linspace(0, 2*np.pi, 100)
    plt.plot(Re*np.cos(angle), Re*np.sin(angle), 'r--', label='Earth Surface', alpha=0.5)
    
    plt.title(f"Strapdown Navigation (24h)\nRadial Drift: {drift:.4f} m")
    plt.xlabel("ECI X (m)")
    plt.ylabel("ECI Y (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig('strapdown_circle.png')
    plt.show()

if __name__ == "__main__":
    solve_strapdown_stable()