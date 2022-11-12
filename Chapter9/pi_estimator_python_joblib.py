import os
import random
import sys
import time
from multiprocessing import Pool
from joblib import Parallel, delayed

def estimate_nbr_points_in_quarter_circle(nbr_estimates):
    """
    Monte Carlo estimate of the number of points in a quarter cirlce using pure python
    :param nbr_estimates:
    :return:
    """
    print(f"executing estimate_nbr_points_in_quarter_circle with {nbr_estimates} on pid: {os.getpid()}")
    s_time = time.time()
    nbr_estimates_in_quarter_unit_circle = 0
    for step in range(int(nbr_estimates)):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        is_in_unit_circle = x * x + y * y <= 1.0
        nbr_estimates_in_quarter_unit_circle += is_in_unit_circle
    print(f"finished processing {nbr_estimates} darts at process#{os.getpid()} in {time.time() - s_time}")
    return nbr_estimates_in_quarter_unit_circle


def joblib_main(nbr_parallel_blocks):
    print(f"parent id: {os.getpid()}")
    nbr_samples_in_total = 1e8
    pool = Pool(processes=nbr_parallel_blocks)
    nbr_samples_per_worker = nbr_samples_in_total / nbr_parallel_blocks
    n_cores = os.cpu_count()
    print(f"n_cores: {n_cores}")
    n_available_cores = len(os.sched_getaffinity(0))
    print(f"n_available_cores: {n_available_cores}")
    print(f"making {nbr_samples_per_worker} samples per {nbr_parallel_blocks} workers")
    # nbr_trials_per_process = [nbr_samples_per_worker] * nbr_parallel_blocks
    t1 = time.time()
    nbr_in_quarter_unit_circles = Parallel(n_jobs=nbr_parallel_blocks, verbose=1)(delayed(estimate_nbr_points_in_quarter_circle)(nbr_samples_per_worker) for sample_idx in range(nbr_parallel_blocks))
    pi_estimate = sum(nbr_in_quarter_unit_circles) * 4 / float(nbr_samples_in_total)
    print(f"estimated pi: {pi_estimate}")
    print(f"Delta: {time.time() - t1}")


if __name__ == '__main__':
    try:
        nbr_parallel_blocks = int(sys.argv[1])
    except Exception as err:
        nbr_parallel_blocks = 2
    nbr_samples_in_total = 1e8
    joblib_main(nbr_parallel_blocks)

# TODO: Why speedup on 4 workers same as 8 workers
