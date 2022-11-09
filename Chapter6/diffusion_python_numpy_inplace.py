import time

import numpy as np
from numpy import (roll, zeros)

grid_shape = (640, 640)


# @profile
def laplacian(grid, out):
    np.copyto(grid, out)
    out *= -4
    out += np.roll(grid, +1, 0)
    out += np.roll(grid, -1, 0)
    out += np.roll(grid, +1, 1)
    out += np.roll(grid, -1, 1)
    # return roll(grid, +1, 0) + roll(grid, -1, 0) + roll(grid, +1, 1) + roll(grid, -1, 1) - 4 * grid


def evolve(grid, dt, out, D=1):
    laplacian(grid, out)
    out *= D * dt
    out += grid


def run_experiment(num_iterations):
    # setting initial conditions
    next_grid = zeros(grid_shape)
    grid = zeros(grid_shape)

    # Drop a dye in the middle
    block_low = int(grid_shape[0] * 0.4)
    block_high = int(grid_shape[0] * 0.5)
    grid[block_low:block_high, block_low:block_high] = 0.005

    # evolve initial conditions
    start = time.time()
    for i in range(num_iterations):
        evolve(grid, 0.1, next_grid)
        grid, next_grid = next_grid, grid
    return time.time() - start


if __name__ == '__main__':
    run_experiment(num_iterations=500)
