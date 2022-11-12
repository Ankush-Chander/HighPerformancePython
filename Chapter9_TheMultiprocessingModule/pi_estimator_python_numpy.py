import os
import sys
import time
import numpy as np
from joblib import Parallel, delayed


def estimate_nbr_points_in_quarter_circle(nbr_estimates):
    """
    Monte Carlo estimate of the number of points in a quarter cirlce using vectorized numpy arrays
    :param nbr_estimates:
    :return:
    """
    # print(f"executing estimate_nbr_points_in_quarter_circle with {nbr_estimates} on pid: {os.getpid()}")
    nbr_estimates = int(nbr_estimates)
    np.random.seed()  # remember to set the seed per process
    xs = np.random.uniform(0, 1, nbr_estimates)
    ys = np.random.uniform(0, 1, nbr_estimates)
    s_time = time.time()
    estimates_in_quarter_unit_circle = (xs * xs + ys * ys) <= 1
    nbr_estimates_in_quarter_unit_circle = sum(estimates_in_quarter_unit_circle)
    print(f"finished processing {nbr_estimates} darts at process#{os.getpid()} in {time.time() - s_time}")
    return nbr_estimates_in_quarter_unit_circle


def numpy_main(nbr_parallel_blocks):
    print(f"parent id: {os.getpid()}")
    nbr_samples_in_total = 1e8
    nbr_samples_per_worker = nbr_samples_in_total / nbr_parallel_blocks
    n_cores = os.cpu_count()
    print(f"n_cores: {n_cores}")
    n_available_cores = len(os.sched_getaffinity(0))
    print(f"n_available_cores: {n_available_cores}")
    print(f"making {nbr_samples_per_worker} samples per {nbr_parallel_blocks} workers")
    # nbr_trials_per_process = [nbr_samples_per_worker] * nbr_parallel_blocks
    t1 = time.time()
    nbr_in_quarter_unit_circles = Parallel(n_jobs=nbr_parallel_blocks, verbose=1)(
        delayed(estimate_nbr_points_in_quarter_circle)(nbr_samples_per_worker) for sample_idx in
        range(nbr_parallel_blocks))
    pi_estimate = sum(nbr_in_quarter_unit_circles) * 4 / float(nbr_samples_in_total)
    print(f"estimated pi: {pi_estimate}")
    print(f"Delta: {time.time() - t1}")


if __name__ == '__main__':
    try:
        nbr_parallel_blocks = int(sys.argv[1])
    except Exception as err:
        nbr_parallel_blocks = 2
    nbr_samples_in_total = 1e8
    numpy_main(nbr_parallel_blocks)

# $ python3 -u pi_estimator_python_numpy.py 1
# making 100000000.0 samples per 1 workers
# [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
# finished processing 100000000 darts at process#958735 in 8.005767107009888
# [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:   10.0s finished
# estimated pi: 3.14193096
# Delta: 10.018234014511108

# $ python3 -u pi_estimator_python_numpy.py 2
# making 50000000.0 samples per 2 workers
# [Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
# finished processing 50000000 darts at process#957551 in 4.806213855743408
# finished processing 50000000 darts at process#957550 in 4.816455841064453
# [Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    6.1s remaining:    0.0s
# [Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    6.1s finished
# estimated pi: 3.1416438
# Delta: 6.151240348815918

# $ python3 -u pi_estimator_python_numpy.py 4
# making 25000000.0 samples per 4 workers
# [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
# finished processing 25000000 darts at process#958077 in 2.8630547523498535
# finished processing 25000000 darts at process#958078 in 3.0352954864501953
# [Parallel(n_jobs=4)]: Done   2 out of   4 | elapsed:    4.0s remaining:    4.0s
# finished processing 25000000 darts at process#958079 in 2.7857375144958496
# finished processing 25000000 darts at process#958076 in 3.000296115875244
# [Parallel(n_jobs=4)]: Done   4 out of   4 | elapsed:    4.4s finished
# estimated pi: 3.14128716
# Delta: 4.450907230377197


# $ python3 -u pi_estimator_python_numpy.py 6
# making 16666666.666666666 samples per 6 workers
# [Parallel(n_jobs=6)]: Using backend LokyBackend with 6 concurrent workers.
# finished processing 16666666 darts at process#958314 in 2.141535520553589
# finished processing 16666666 darts at process#958312 in 2.158376693725586
# [Parallel(n_jobs=6)]: Done   2 out of   6 | elapsed:    3.2s remaining:    6.4s
# finished processing 16666666 darts at process#958313 in 2.980018138885498
# finished processing 16666666 darts at process#958310 in 2.963810682296753
# finished processing 16666666 darts at process#958311 in 3.3091647624969482
# finished processing 16666666 darts at process#958309 in 3.231931209564209
# [Parallel(n_jobs=6)]: Done   6 out of   6 | elapsed:    4.3s finished
# estimated pi: 3.14142916
# Delta: 4.316645622253418


# $ python3 -u pi_estimator_python_numpy.py 8
# making 12500000.0 samples per 8 workers
# [Parallel(n_jobs=8)]: Using backend LokyBackend with 8 concurrent workers.
# finished processing 12500000 darts at process#958478 in 2.900470495223999
# finished processing 12500000 darts at process#958474 in 3.12073016166687
# [Parallel(n_jobs=8)]: Done   2 out of   8 | elapsed:    4.1s remaining:   12.4s
# finished processing 12500000 darts at process#958477 in 3.1272671222686768
# finished processing 12500000 darts at process#958479 in 3.063809394836426
# finished processing 12500000 darts at process#958475 in 3.160606622695923
# finished processing 12500000 darts at process#958472 in 3.116447687149048
# finished processing 12500000 darts at process#958476 in 2.943378210067749
# finished processing 12500000 darts at process#958473 in 3.2473549842834473
# [Parallel(n_jobs=8)]: Done   8 out of   8 | elapsed:    4.3s finished
# estimated pi: 3.14163572
# Delta: 4.328490972518921


# TODO: Why speedup on 8 workers same/less than 4 workers
