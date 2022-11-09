import time

from numpy import (roll, zeros)

grid_shape = (640, 640)

@profile
def laplacian(grid):
    return roll(grid, +1, 0) + roll(grid, -1, 0) + roll(grid, +1, 1) + roll(grid, -1, 1) - 4 * grid

@profile
def evolve(grid, dt, D=1):
    return grid + dt * D * laplacian(grid)


def run_experiment(num_iterations):
    # setting initial conditions
    grid = zeros(grid_shape)

    # Drop a dye in the middle
    block_low = int(grid_shape[0] * 0.4)
    block_high = int(grid_shape[0] * 0.5)
    grid[block_low:block_high, block_low:block_high] = 0.005

    # evolve initial conditions
    start = time.time()
    for i in range(num_iterations):
        grid = evolve(grid, 0.1)

    return time.time() - start

if __name__ == '__main__':
    run_experiment(num_iterations=500)