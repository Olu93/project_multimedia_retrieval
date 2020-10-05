

def exception_catcher(func):
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {f"ERR_{func.__name__}": (str(type(e)), str(e))}
    return new_func