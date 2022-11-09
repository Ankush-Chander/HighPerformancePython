
import time
import traceback

from icecream import ic
from line_profiler import LineProfiler

lp = LineProfiler()
grid_shape = (640, 640)

# profile is added in builtin namespace when run via kernprof
# usage:  kernprof -l -v diffusion_python.py
@profile
def evolve(grid, dt, D=1.0):
    xmax, ymax = grid_shape
    new_grid = [[0.0] * ymax for x in range(xmax)]

    for i in range(xmax):
        for j in range(ymax):
            try:

                # ic(grid)
                grid_xx = (grid[(i + 1) % xmax][j] + grid[(i - 1) % xmax][j] - 2.0 * grid[i][j])
                grid_yy = (grid[i][(j + 1) % ymax] + grid[i][(j - 1) % ymax] - 2.0 * grid[i][j])
                new_grid[i][j] = grid[i][j] + D * (grid_xx + grid_yy) * dt
            except Exception as err:
                print(traceback.format_exc())
                # ic((i + 1) % xmax)
                # ic(grid)
                return
    return new_grid


def run_experiment(num_iterations):
    # setting initial conditions
    xmax, ymax = grid_shape
    grid = [[0.0] * ymax for x in range(xmax)]
    ic(len(grid))
    # Drop a dye in the middle
    block_low = int(grid_shape[0] * 0.4)
    block_high = int(grid_shape[0] * 0.5)
    for x in range(block_low, block_high):
        for y in range(block_low, block_high):
            grid[x][y] = 0.005

    # evolve initial conditions
    start = time.time()
    # ic(grid)
    for i in range(num_iterations):
        ic(i)
        ic(type(grid))
        grid = evolve(grid, 0.1)

    return time.time() - start


if __name__ == '__main__':
    run_experiment(num_iterations=100)
