import cProfile

def parallel_profile(func):
    """
    Use this to profile parallel workers
    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profiler.print_stats(sort="cumtime")
        return result
    return wrapper