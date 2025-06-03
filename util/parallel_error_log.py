import os
import traceback

def error_logger(func):
    """Decorator to log errors to a file defined in the class property."""

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            log_file = getattr(self, "log_file_name", "default_log.txt")

            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Write to log file
            with open(log_file, "a") as f:
                f.write(f"Error in {func.__name__}: {str(e)}\n")
                f.write(traceback.format_exc() + "\n")

            print(f"Error logged in {log_file}")

    return wrapper
