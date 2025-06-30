'''Numpy, scipy and sklearn background libraries spawn threads that may clog
CPU through oversubscription. This can interfere with the performance 
measurement of algorithms that are configured, especially if all system cores
are utilized in the tournaments. Because of this, we set the maximum thread
numbers to 1. It has to happen before any of the modules are loaded.'''

import os


def set_background_thread_nr() -> None:
    '''
    Set thread number of background libraries for numpy, scipy, sklearn to 1.

    Returns
    -------
    None
    '''

    # Set max threads for various backends
    os.environ["OMP_NUM_THREADS"] = "1"         # OpenMP
    os.environ["OPENBLAS_NUM_THREADS"] = "1"    # OpenBLAS
    os.environ["MKL_NUM_THREADS"] = "1"         # MKL (Intel Math Kernel Lib)
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate
    os.environ["NUMEXPR_NUM_THREADS"] = "1"     # NumExpr

    # Optional: Limit threadpool (used by joblib, e.g. in sklearn)
    os.environ["JOBLIB_START_METHOD"] = "fork"  # safer parallelism on Unix
