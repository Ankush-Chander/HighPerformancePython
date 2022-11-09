import os
import random
import sys
import time
from multiprocessing import Pool


def estimate_nbr_points_in_quarter_circle(nbr_estimates):
    """
    Monte Carlo estimate of the number of points in a quarter cirlce using pure python
    :param nbr_estimates:
    :return:
    """
    print(f"executing estimate_nbr_points_in_quarter_circle with {nbr_estimates} on pid: {os.getpid()}")
    nbr_estimates_in_quarter_unit_circle = 0
    for step in range(int(nbr_estimates)):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        is_in_unit_circle = x * x + y * y <= 1.0
        nbr_estimates_in_quarter_unit_circle += is_in_unit_circle
    return nbr_estimates_in_quarter_unit_circle


if __name__ == '__main__':
    try:
        nbr_parallel_blocks = int(sys.argv[1])
    except Exception as err:
        nbr_parallel_blocks = 2
    nbr_samples_in_total = 1e8

    pool = Pool(processes=nbr_parallel_blocks)
    nbr_samples_per_worker = nbr_samples_in_total / nbr_parallel_blocks
    print(f"making {nbr_samples_per_worker} samples per {nbr_parallel_blocks} workers")
    nbr_trials_per_process = [nbr_samples_per_worker] * nbr_parallel_blocks
    t1 = time.time()
    nbr_in_quarter_unit_circles = pool.map(estimate_nbr_points_in_quarter_circle, nbr_trials_per_process)
    pi_estimate = sum(nbr_in_quarter_unit_circles) * 4 / float(nbr_samples_in_total)
    print(f"estimated pi: {pi_estimate}")
    print(f"Delta: {time.time() - t1}")

# making 100000000.0 samples per 1 workers
# executing estimate_nbr_points_in_quarter_circle with 100000000.0 on pid: 972058
# estimated pi: 3.1412182
# Delta: 51.92072415351868

# making 50000000.0 samples per 2 workers
# executing estimate_nbr_points_in_quarter_circle with 50000000.0 on pid: 973456
# executing estimate_nbr_points_in_quarter_circle with 50000000.0 on pid: 973457
# estimated pi: 3.14174644
# Delta: 24.984251499176025

# making 33333333.333333332 samples per 3 workers
# executing estimate_nbr_points_in_quarter_circle with 33333333.333333332 on pid: 974451
# executing estimate_nbr_points_in_quarter_circle with 33333333.333333332 on pid: 974453
# executing estimate_nbr_points_in_quarter_circle with 33333333.333333332 on pid: 974452
# estimated pi: 3.1414508
# Delta: 16.808180332183838
#
# making 25000000.0 samples per 4 workers
# executing estimate_nbr_points_in_quarter_circle with 25000000.0 on pid: 975371
# executing estimate_nbr_points_in_quarter_circle with 25000000.0 on pid: 975372
# executing estimate_nbr_points_in_quarter_circle with 25000000.0 on pid: 975373
# executing estimate_nbr_points_in_quarter_circle with 25000000.0 on pid: 975374
# estimated pi: 3.14181176
# Delta: 12.86291241645813


# making 16666666.666666666 samples per 6 workers
# executing estimate_nbr_points_in_quarter_circle with 16666666.666666666 on pid: 976153
# executing estimate_nbr_points_in_quarter_circle with 16666666.666666666 on pid: 976154
# executing estimate_nbr_points_in_quarter_circle with 16666666.666666666 on pid: 976155
# executing estimate_nbr_points_in_quarter_circle with 16666666.666666666 on pid: 976156
# executing estimate_nbr_points_in_quarter_circle with 16666666.666666666 on pid: 976157
# executing estimate_nbr_points_in_quarter_circle with 16666666.666666666 on pid: 976158
# estimated pi: 3.14125152
# Delta: 14.648477554321289


# making 12500000.0 samples per 8 workers
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 977013
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 977014
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 977016
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 977015
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 977017
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 977019
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 977018
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 977020
# estimated pi: 3.14171596
# Delta: 14.694321393966675
#
# Process finished with exit code 0


# TODO: Why speedup on 4 workers same as 8 workers