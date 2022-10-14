import asyncio
import functools
import inspect


class SyncClassGenerationMetaClass(type):
    """
        Generates a class with synchronous methods from a class that has some async methods.
        Example use:
            class SynchronousClass(AsyncClass, metaclass=SyncClassGenerationMetaClass):
                pass
    """
    @staticmethod
    def _run_async_synchronously(function):
        @functools.wraps(function)
        def _run_synchronously(*args, **kwargs):
            return asyncio.run(function(*args, **kwargs))
        return _run_synchronously

    def __new__(cls, name, bases, namespace):
        for base in bases:
            for attribute_name in dir(base):
                attribute_value = getattr(base, attribute_name)
                if inspect.iscoroutinefunction(attribute_value):
                    namespace[attribute_name] = cls._run_async_synchronously(attribute_value)
        return super().__new__(cls, name, bases, namespace)
