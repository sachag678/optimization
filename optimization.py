#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def rosenbrock(x, y, a = 1, b = 100):
    return (a - x)**2 + b * (y - x**2)**2

def parabaloid(x, y):
    return (10*x**2 + y**2)/2

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

params = {
    "Rosenbrock": {
        "x_min":-2,
        "x_max":2,
        "y_min":-1,
        "y_max":3,
        "init_x": -1.2,
        "init_y": 1
    },
    "Paraboloid": {
        "x_min":-4,
        "x_max":4,
        "y_min":-4,
        "y_max":4,
        "init_x": 3,
        "init_y": 3
    },
    "Himmelblau": {
        "x_min":-5,
        "x_max":5,
        "y_min":-5,
        "y_max":5,
        "init_x": 0,
        "init_y": 0
    }
}

def plot2d(f, xs, ys, animate=False):
    x = np.linspace(-2, 2, 10000)
    y = np.linspace(-1, 3, 10000)
    xv, yv = np.meshgrid(x, y)
    z = np.log10(f(xv, yv))

    fig, ax = plt.subplots()
    ax.imshow(z, aspect='equal', origin='lower', extent = (-2, 2, -1, 3))
    ax.contour(xv, yv, z, colors = 'white', levels = 7, linewidths=0.5)

    if animate:
        def animate(i):
            ax.plot(xs[:i], ys[:i], 'rx--', linewidth=0.05)

        ani = FuncAnimation(fig, animate, frames=len(xs) - 1, interval=(10000 // len(xs)), repeat=False)
    else:
        ax.plot(xs, ys, 'rx--', linewidth=0.1)

    return fig

def plot3d(f, xs, ys, animate=False):
    x = np.linspace(-2, 2, 10000)
    y = np.linspace(-1, 3, 10000)
    xv, yv = np.meshgrid(x, y)
    z = np.log10(f(xv, yv))
    zs = np.log10(f(np.array(xs), np.array(ys)))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xv, yv, z, cmap=plt.cm.magma)
    if animate:
        def animate(i):
            ax.plot(xs[:i], ys[:i], zs[:i], 'ro--')

        ani = FuncAnimation(fig, animate, frames=len(xs) - 1, interval=(10000 // len(xs)), repeat=False)
    else:
        ax.plot(xs, ys, zs, 'ro--')
    plt.show()

def dellfy(x, y):
    return 200 * (y - x**2)

def dellfx(x, y):
    return 400 * x**3 - 400 * x * y + 2 * x - 2

def delf(x, y):
    return np.array([dellfx(x, y), dellfy(x, y)])

def hessian(x, y):
    return np.array([[1200 * x**2 - 400 * y + 2, -400 * x], [-400 * x, 200]])

def approx_hessian(f, x, y, h = 0.000001, k = 0.000001):
    fxy = (f(x + h, y + k) - f(x + h, y - k) - f(x - h, y + k) + f(x - h, y - k)) / (4 * h * k)
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2
    fyy = (f(x, y + k) - 2 * f(x, y) + f(x, y - k)) / k**2
    return np.array([[fxx , fxy], [fxy, fyy]])

def approx_deriv_x(f, x, y, h = 0.000001):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

def approx_deriv_y(f, x, y, h = 0.000001):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)

def backtracking_line_search(f, df, px, py, x, y, alpha = 1, c = 0.01, rho= 0.75):
    while f(x + alpha * px, y + alpha * py) > f(x, y) + alpha * c * (df[0] * px + df[1] * py):
        alpha *= rho
    return alpha

def fixed_step_size(alpha):
    return alpha

# uses del f
def steepest_descent(f, x0=1.2, y0=1.2, c=0.01, rho=0.75):
    xs = [x0]
    ys = [y0]
    x = x0
    y = y0
    count = 0
    while True:
        px = -approx_deriv_x(f, x, y)
        py = -approx_deriv_y(f, x, y)
        step_size = backtracking_line_search(f, np.array([-px, -py]), px, py, x, y, c=c, rho=rho)
        print('alpha: {}, p: ({}, {})'.format(step_size, px, py))
        old_x = x
        old_y = y
        x += step_size * px
        y += step_size * py
        xs.append(x)
        ys.append(y)
        #print('x: {}, y: {}'.format(x, y))
        if ((old_x - x)**2 + (old_y - y)**2)**0.5 < 0.0001:
            break
        count += 1
    print('steepest_descent: ', x, y, count)
    return xs, ys


# uses Hessian
def newtons_method(f, x0=1.2, y0=1.2, beta=0.001, multiple_identity=False, flip_negative_eigs=False, c=0.01, rho=0.75):
    xs = [x0]
    ys = [y0]
    x = x0
    y = y0
    count = 0
    while True:
        df = np.array([approx_deriv_x(f, x, y), approx_deriv_y(f, x, y)])
        H = approx_hessian(f, x, y)
        eigs = np.linalg.eigvals(H)
        if multiple_identity:
            if min(eigs) > 0:
                tau = 0
            else:
                tau = - min(eigs) + beta
            Hbar = H + np.eye(2) * tau # modifies using multiples of identity
        elif flip_negative_eigs:
            mod_mat = (H>0).astype(int)
            mod_mat[mod_mat==0] = -1
            Hbar = H * mod_mat
        else:
            Hbar = H
        p = -np.linalg.inv(Hbar) @ df
        px = p[0]
        py = p[1]
        step_size = backtracking_line_search(f, df, px, py, x, y, c=c, rho=rho)
        print('alpha: {}, p: ({}, {})'.format(step_size, px, py))
        old_x = x
        old_y = y
        x += step_size * px
        y += step_size * py
        xs.append(x)
        ys.append(y)
        #print('x: {}, y: {}'.format(x, y))
        #if ((old_x - x)**2 + (old_y - y)**2)**0.5 < 0.0001:
        if np.linalg.norm(df, 2) < 0.00001 or ((old_x - x)**2 + (old_y - y)**2)**0.5 < 0.0001:
            break
        count += 1
    print('newtons method: ',  x, y, count)
    return xs, ys

# uses Bk
def bfgs():
    pass

if __name__ == '__main__':
    xs, ys = steepest_descent(rosenbrock, x0=-1.2, y0=1)
    plot2d(rosenbrock, xs, ys)
    xs, ys = newtons_method(rosenbrock, x0=-1.2, y0=1)
    plot3d(rosenbrock, xs, ys, animate=True)
