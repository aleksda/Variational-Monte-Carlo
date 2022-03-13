import functools
import platform
import sys
import time

import numpy

print("PLATFORM:\n" + platform.platform())
print("\nVERSION:\nPython " + sys.version)
print("numpy version: {}".format(numpy.__version__))
print("\nBENCHMARKS:")


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer
