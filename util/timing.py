import time

class TicToc:

    @classmethod
    def tic(cls):
        cls.start_time = time.perf_counter_ns()

    @classmethod
    def toc(cls):
        end_time = time.perf_counter_ns()
        return (end_time - cls.start_time) / 10**9