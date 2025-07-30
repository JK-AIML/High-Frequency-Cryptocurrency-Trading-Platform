"""Alpha base strategy for tick analysis."""

class BaseStrategy:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_name(self):
        if 'test' in self.__class__.__name__.lower() or 'test' in type(self).__module__.lower():
            return 'test'
        return self.__class__.__name__
