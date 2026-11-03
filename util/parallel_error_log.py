import os
import traceback
from typing import Any, Callable, TypeVar, cast

FunctionType = TypeVar("FunctionType", bound=Callable[..., Any])


def error_logger(func: FunctionType) -> FunctionType:
    """Decorator to log errors to a file defined in the class property, then re-raise them so the
    process exits with a non-zero exit code."""

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            log_file = getattr(self, "log_file_name", "default_log.txt")

            # Ensure directory exists
            if os.path.dirname(log_file):
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Write to log file
            with open(log_file, "a") as f:
                f.write(f"Error in {func.__name__}: {str(e)}\n")
                f.write(traceback.format_exc() + "\n")

            print(f"Error logged in {log_file}")
            raise

    return cast(FunctionType, wrapper)
