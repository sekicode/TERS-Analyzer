import copy


class Repository:
    def __init__(self) -> None:
        self.__data = None

    @property
    def data(self):
        return self.__data

    def set_data(self, data):
        self.__data = copy.deepcopy(data)

