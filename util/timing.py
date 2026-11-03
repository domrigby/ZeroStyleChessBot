import time


class TicToc:
    start_time: int

    @classmethod
    def tic(cls) -> None:
        cls.start_time = time.perf_counter_ns()

    @classmethod
    def toc(cls) -> float:
        end_time = time.perf_counter_ns()
        return (end_time - cls.start_time) / 10**9
