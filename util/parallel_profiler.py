import cProfile
from typing import Any, Callable, TypeVar, cast

FunctionType = TypeVar("FunctionType", bound=Callable[..., Any])


def parallel_profile(func: FunctionType) -> FunctionType:
    """
    Use this to profile parallel workers
    :param func:
    :return:
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profiler.print_stats(sort="tottime")
        return result

    return cast(FunctionType, wrapper)
