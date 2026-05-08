import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def rosenbrock(x):
    """
    The Rosenbrock function (The Banana Function).
    f(x, y) = 100*(y - x^2)^2 + (1 - x)^2
    Global minimum is at (1, 1) where f = 0.
    """
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def optimize_rosenbrock():
    # 1. Initial Guess (Standard challenging starting point)
    x0 = np.array([-1.2, 1.0])
    
    # Storage for the path taken by the optimizer
    path = [x0]
    
    # Callback function to record the path
    def callback(xk):
        path.append(xk)
        
    print(f"Starting Optimization from: {x0}")
    
    # 2. Run Optimization using BFGS (quasi-Newton method)
    # This is a standard gradient-based method for unconstrained optimization
    res = minimize(rosenbrock, x0, method='BFGS', callback=callback, tol=1e-6)
    
    # 3. Print Results
    print("\n--- Optimization Results ---")
    print(f"Success: {res.success}")
    print(f"Iterations: {res.nit}")
    print(f"Found Minimum at (x, y): {res.x}")
    print(f"Function Value: {res.fun}")
    print(f"True Minimum is at (1.0, 1.0)")

    # 4. Plotting the Result (Contour Plot)
    path = np.array(path)
    
    # Generate grid for contour plot
    x_range = np.linspace(-2, 2, 400)
    y_range = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 100 * (Y - X**2)**2 + (1 - X)**2
    
    plt.figure(figsize=(8, 6))
    
    # Logarithmic scale for contours to see the valley better
    plt.contourf(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    plt.colorbar(label='Objective Function (Log Scale)')
    
    # Plot the path
    plt.plot(path[:, 0], path[:, 1], 'w.-', label='Optimization Path')
    plt.plot(1, 1, 'r*', markersize=15, label='Global Min (1,1)')
    plt.plot(x0[0], x0[1], 'rx', markersize=10, label='Start')
    
    plt.title('Minimizing Rosenbrock Function (BFGS Method)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot for the report
    plt.savefig('optimization_result.png')
    plt.show()

if __name__ == "__main__":
    optimize_rosenbrock()