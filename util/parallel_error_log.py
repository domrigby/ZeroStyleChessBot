import traceback

def error_logger(func):
    """Decorator to log errors to a file defined in the class property."""

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if hasattr(self, "log_file_name"):
                log_file = self.log_file_name  # Get filename from the class property
            else:
                log_file = "default_log.txt"  # Fallback filename

            with open(log_file, "a") as f:
                f.write(f"Error in {func.__name__}: {str(e)}\n")
                f.write(traceback.format_exc() + "\n")

            print(f"Error logged in {log_file}")

    return wrapper
