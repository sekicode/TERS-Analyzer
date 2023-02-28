import numpy as np

from core_math import Math
from core_winspec import SpeFile


class MySpeFile(SpeFile):
    def __init__(self, name):
        super().__init__(name)

    def get_time_from_spefile(self):
        return Math.get_time_from_spefile(self)

    def get_timezone_from_spefile(self):
        return Math.get_timezone_from_spefile(self)

    def get_info_from_spefile(self):
        return Math.get_info_from_spefile(self)

    def read_winspec_file(self):
        return Math.read_winspec_file(self)

    def get_raw_spectra_from_spefile(self):
        return Math.get_raw_spectra_from_spefile(self)
