from core_error import CustomError
from core_logger import logger


class MyWrapper():
    def __init__(self) -> None:
        pass

    def safe(func):
        def wrapper(self, *args, **kargs):
            try:
                func_name = func.__name__
                return func(self, *args, **kargs)
            except Exception as err:
                custom_err = CustomError(error=err, method=func_name)
                custom_err.debug()
                custom_err.show_error()
        return wrapper

    def info(func):
        def wrapper(self, *args, **kargs):
            func_name = func.__name__
            silence = kargs.get('silence')
            if not silence:
                logger.info('{} ...'.format(func_name))

            ret = func(self, *args, **kargs)
            if not silence:
                logger.info('{} done'.format(func_name))
            return ret
        return wrapper
