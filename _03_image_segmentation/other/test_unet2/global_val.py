class Seed():
    def __init__(self):
        self._global_dict = 0

    def set_seed(self, value):
        self._global_dict = value

    def get_seed(self, defValue=None):
        try:
            return self._global_dict
        except KeyError:
            return defValue


seed = Seed()


class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance

    def set_seed(self, value):
        self._global_dict = value

    def get_seed(self, defValue=None):
        try:
            return self._global_dict
        except KeyError:
            return defValue