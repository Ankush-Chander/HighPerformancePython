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

# >>> python3 -u pi_estimator_python_joblib.py 1
# making 100000000.0 samples per 1 workers
# [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
# executing estimate_nbr_points_in_quarter_circle with 100000000.0 on pid: 952850
# finished processing 100000000.0 darts at process#952850 in 51.72025775909424
# [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:   51.7s finished
# estimated pi: 3.1414128
# Delta: 51.72098112106323


# >>> python3 -u pi_estimator_python_joblib.py 4
# making 25000000.0 samples per 4 workers
# [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
# executing estimate_nbr_points_in_quarter_circle with 25000000.0 on pid: 952661
# executing estimate_nbr_points_in_quarter_circle with 25000000.0 on pid: 952659
# executing estimate_nbr_points_in_quarter_circle with 25000000.0 on pid: 952658
# executing estimate_nbr_points_in_quarter_circle with 25000000.0 on pid: 952660
# finished processing 25000000.0 darts at process#952658 in 14.697245597839355
# finished processing 25000000.0 darts at process#952659 in 14.770778894424438
# [Parallel(n_jobs=4)]: Done   2 out of   4 | elapsed:   15.1s remaining:   15.1s
# finished processing 25000000.0 darts at process#952661 in 14.849751472473145
# finished processing 25000000.0 darts at process#952660 in 14.862541437149048
# [Parallel(n_jobs=4)]: Done   4 out of   4 | elapsed:   15.2s finished
# estimated pi: 3.14120364
# Delta: 15.169500350952148


# >>> python3 -u pi_estimator_python_joblib.py 8
# making 12500000.0 samples per 8 workers
# [Parallel(n_jobs=8)]: Using backend LokyBackend with 8 concurrent workers.
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 953420
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 953415
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 953416
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 953421
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 953417
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 953418
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 953419
# executing estimate_nbr_points_in_quarter_circle with 12500000.0 on pid: 953422
# finished processing 12500000.0 darts at process#953417 in 16.571314811706543
# finished processing 12500000.0 darts at process#953415 in 16.9799485206604
# [Parallel(n_jobs=8)]: Done   2 out of   8 | elapsed:   17.4s remaining:   52.1s
# finished processing 12500000.0 darts at process#953422 in 17.139020204544067
# finished processing 12500000.0 darts at process#953421 in 17.255162000656128
# finished processing 12500000.0 darts at process#953420 in 17.42788004875183
# finished processing 12500000.0 darts at process#953419 in 17.40841555595398
# finished processing 12500000.0 darts at process#953416 in 17.614360332489014
# finished processing 12500000.0 darts at process#953418 in 17.581160306930542
# [Parallel(n_jobs=8)]: Done   8 out of   8 | elapsed:   18.0s finished
# estimated pi: 3.14180008
# Delta: 18.03565788269043

# TODO: Why speedup on 8 workers same/less than 4 workers
