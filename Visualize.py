import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.animation import FuncAnimation

from Functions import FunctionFactory
from PSO import PSO
import argparse

np.random.seed(4906)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('f_name')
    parser.add_argument('--clarity', dest='clarity', type=int, default=100)
    parser.add_argument('--iteration', dest='iteration', type=int, default=30)
    parser.add_argument('--population', dest='population', type=int, default=20)
    parser.add_argument('--w', dest='w', type=float, default=0.5)
    parser.add_argument('--rp', dest='rp', type=float, default=1.25)
    parser.add_argument('--rg', dest='rg', type=float, default=2.0)
    parser.add_argument('--interval', dest='interval', type=int, default=500)
    parser.add_argument('--size', dest='size',type=float, default=5.0)
    # parser.add_argument('--arrow', dest='arrow', type=bool, default=False)

    args = parser.parse_args()
    function_name = args.f_name
    function_factory = FunctionFactory()
    (x_low, x_high, y_low, y_high), func = function_factory.get_function(function_name)
    if func is None:
        print("No such function")
        sys.exit(0)
    clarity = args.clarity
    x = np.linspace(x_low, x_high, clarity)
    y =  np.linspace(y_low, y_high, clarity)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func((X[i][j], Y[i][j])) for i in range(clarity) for j in range(clarity)]).reshape(clarity, clarity)


    fig, ax = plt.subplots(figsize=(10, 6))
    if np.max(np.abs(Z)) >= 500:
        contour = ax.contourf(X, Y, Z, levels=20,  norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()), cmap='viridis')
    else:
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    fig.colorbar(contour)

    iteration = args.iteration
    population = args.population
    w = args.w
    rp = args.rp
    rg = args.rg
    # with_arrow = args.arrow

    pso = PSO(iteration, population, w, rp, rg, x_low, x_high, y_low, y_high, func)
    scatter = ax.scatter(pso.positions[:, 0], pso.positions[:, 1], s=args.size, c='red')
    arrows = None

    def update(frame):
        pso.solve()
        new_pos = np.column_stack((pso.positions[:, 0], pso.positions[:, 1]))
        scatter.set_offsets(new_pos)

        # nonlocal arrows
        # if arrows is not None:
        #     arrows.remove()
        # arrows = ax.quiver(pso.positions[:, 0], pso.positions[:, 1], pso.velocities[:, 0], pso.velocities[:, 1],
        #                    color='black', scale=50, width=1e-3)
        # print(frame)
        if frame == iteration - 1:
            plt.close()
        return scatter, arrows

    animation = FuncAnimation(fig, update, frames=iteration, interval=args.interval)

    plt.show()
    x_best, y_best = pso.get_solution()
    print(f'The global minimum for {function_name} is {func((x_best, y_best))} at ({x_best, y_best})')

if __name__ == '__main__':
    main()