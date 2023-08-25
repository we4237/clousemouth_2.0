import time
from functools import wraps


def timer(log_str: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = round(end_time - start_time)
            print(f'{log_str} cost: {execution_time}')
            return result

        return wrapper

    return decorator
