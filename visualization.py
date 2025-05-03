import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return 2*x - y/x

def exact_solution(x):
    return (2 * x**2)/3 - 2/(3 * x)

def solved_function_points(x0, y0, n, h):
    points = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n):
        x_new = x + h
        y_new = exact_solution(x_new)
        x, y = x_new, y_new
        points.append((x, y))

    return points

def euler_method(x0, y0, n, h):
    points = [(x0, y0)]
    x, y = x0, y0

    for _ in range(n):
        x_new = x + h
        y_new = y + h * f(x, y)

        x, y = x_new, y_new
        points.append((x, y))

    return points

def improved_euler_method(x0, y0, n, h):
    points = [(x0, y0)]
    x, y = x0, y0

    for _ in range(n):
        x_new = x + h
        p = y + h * f(x, y)

        y_new = y + h/2 * (f(x, y) + f(x_new, p))
        x, y = x_new, y_new
        points.append((x, y))
    return points

def runge_kutta(x0, y0, n, h):
    points = [(x0, y0)]
    x, y = x0, y0

    for _ in range(n):
        x_new = x + h

        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x_new, y + k3)
        y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6

        x, y = x_new, y_new
        points.append((x, y))
    return points

def calculate_error(exact_points, method_points):
    errors = []
    for (_, exact_y), (_, method_y) in zip(exact_points, method_points):
        absolute_error = abs(exact_y - method_y)
        errors.append(absolute_error)
    return errors

def calculate_average_error(exact_points, method_points):
    errors = [abs(exact_y - method_y) for (_, exact_y), (_, method_y) in zip(exact_points, method_points)]
    return sum(errors) / len(errors)

def main():
    x0 = 1
    y0 = 0
    n = 10
    x_final = 5
    h = (x_final - x0) / n

    euler_points = euler_method(x0, y0, n, h)   
    improved_euler_points = improved_euler_method(x0, y0, n, h)
    runge_kutta_points = runge_kutta(x0, y0, n, h)
    function_points = solved_function_points(x0, y0, n, h)

    euler_errors = calculate_error(function_points, euler_points)
    improved_euler_errors = calculate_error(function_points, improved_euler_points)
    runge_kutta_errors = calculate_error(function_points, runge_kutta_points)

    print(f"N = {n}, h = {h}")
    print("Exact Solution Points:", function_points)
    print("Euler Method Points:", euler_points)
    print("Improved Euler Method Points:", improved_euler_points)
    print("Runge-Kutta Method Points:", runge_kutta_points)
    print("Euler Method Errors:", [np.format_float_scientific(err, precision=4) for err in euler_errors])
    print("Improved Euler Errors:", [np.format_float_scientific(err, precision=6) for err in improved_euler_errors])
    print("Runge‑Kutta Errors:", [np.format_float_scientific(err, precision=8) for err in runge_kutta_errors])

    plt.figure(figsize=(10,6))
    plt.plot([point[0] for point in function_points], [point[1] for point in function_points],
             label="Exact Solution", linestyle='-', color='black')
    plt.plot([point[0] for point in euler_points], [point[1] for point in euler_points],
             label="Euler Method", linestyle='-', color='red')
    plt.plot([point[0] for point in improved_euler_points], [point[1] for point in improved_euler_points],
             label="Improved Euler Method", linestyle='-', color='blue')
    plt.plot([point[0] for point in runge_kutta_points], [point[1] for point in runge_kutta_points],
             label="Runge-Kutta Method", linestyle='-', color='green')
    
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)
    
    plt.title("Numerical Methods for Solving ODEs")
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison.pdf")


    plt.figure(figsize=(10, 6))
    n_values = [10, 20, 40, 80]
    step_sizes = [(x_final - x0) / n for n in n_values]

    euler_errors_convergence = []
    improved_euler_errors_convergence = []
    runge_kutta_errors_convergence = []

    for n in n_values:
        h = (x_final - x0) / n

        func_pts = solved_function_points(x0, y0, n, h)
        e_pts = euler_method(x0, y0, n, h)
        ie_pts = improved_euler_method(x0, y0, n, h)
        rk_pts = runge_kutta(x0, y0, n, h)

        euler_errors_convergence.append(calculate_average_error(func_pts, e_pts))
        improved_euler_errors_convergence.append(calculate_average_error(func_pts, ie_pts))
        runge_kutta_errors_convergence.append(round(calculate_average_error(func_pts, rk_pts), 8))

    plt.loglog(step_sizes, euler_errors_convergence, label="Euler Method", marker='o', color='red')
    plt.loglog(step_sizes, improved_euler_errors_convergence, label="Improved Euler Method", marker='o', color='blue')
    plt.loglog(step_sizes, runge_kutta_errors_convergence, label="Runge-Kutta Method", marker='o', color='green')
    plt.legend()

    ax = plt.gca()
    ax.set_xticks(step_sizes)
    ax.minorticks_off()

    plt.title("Error Convergence Plot")
    plt.xlabel("Step Size (h)")
    plt.ylabel("Average Error")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    main()