
from typing import Type
from core_setting import Setting
import numpy as np

from core_IO import TERS_IO
from core_math import Math, TERS_Math
from core_my_winspec import MySpeFile
from core_plot import TERS_Plot
from core_repository import Repository
import os



class TERS:
    def __init__(self, file_cal: str = None, file_ori: list = [], settings: Setting = None, **kargs) -> None:
        if not isinstance(file_ori, list) and not isinstance(file_ori, tuple):
            file_ori = [file_ori]
        self.settings = settings
        self.__repo = Repository()
        self.properties = kargs
        self.properties['filenames'] = file_ori
        self.default_properties = {'pixel_x': 1,
                                   'distance': 0}
        self.data_ori = self.data_cal = self.spefile_cal = self.spefile_ori = None
        self.set_props_from_default(force=False)

        self._read(file_cal, file_ori, **kargs)

        self._create_classes()

    @property
    def data(self):
        return self.__repo.data

    @property
    def repo(self):
        return self.__repo

    def set_data(self, data):
        self.repo.set_data(data)

    def _read(self, file_cal, file_ori, **kargs):
        if kargs.get('data'):
            data = kargs.get('data')
        else:
            self._get_spe_file(file_cal, file_ori)
            self._read_cal(file_cal, **kargs)
            self._read_ori(file_ori, **kargs)
            data = self._calibrate()

        self.repo.set_data(data)

    def _create_classes(self):
        self.math = TERS_Math(self.repo, self.properties)
        self.io = TERS_IO()
        self.plot = TERS_Plot(self.repo, self.math, self.io,
                              self.properties, self.settings)

    def get_color(self, settings):
        color = settings.get('color')
        if not color:
            color = 'winter'
        return color

    def _get_spe_file(self, file_cal=None, file_ori=None):
        if file_cal and file_cal.split('.')[-1] == 'SPE':
            self.spefile_cal = MySpeFile(file_cal)

        if file_ori and file_ori[0].split('.')[-1] == 'SPE':
            self.spefile_ori = MySpeFile(file_ori[0])

    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        else:
            return 0

    @classmethod
    def _read_txt_or_csv(cls, filename: str):
        # Load file by extension
        file_ext = Math.get_extension_from_filename(filename).lower()
        try:
            if file_ext == ".csv":
                return np.loadtxt(filename, delimiter=',')
            elif file_ext == ".txt":
                return np.loadtxt(filename)
        except:
            pass

        # Try different delimiters
        for deli in [' ', '', ',']:
            try:
                return np.loadtxt(filename, delimiter=deli)
            except:
                pass

        # Nothing works. Fail it
        raise TypeError

    def _read_cal(self, file=None, **kargs):
        if file is None or file == '':
            return None

        if file.split('.')[-1] == 'SPE':
            self.data_cal = self.spefile_cal.get_raw_spectra_from_spefile()[0]
            # cyclo = self.math.get_spectra_from_spefile(SpeFile(file))[0]
            laser_wavelength = kargs.get('laser_wavelength')
            method = kargs.get('method')
            self.data_cal = self._calibrate_from_ref(method="ref cm-1",
                                                     fitting_method=method,
                                                     laser_wavelength=laser_wavelength,
                                                     ref_chemical='cyclo')

        else:
            self.data_cal = self._read_txt_or_csv(file)

    def _read_ori(self, files=None, **kargs):
        if not files or files == '':
            return None

        self.data_ori = []
        if len(files) == 1 and files[0].split('.')[-1] == 'SPE':
            # self.data_ori = self.math.get_spectra_from_spefile(
            #     SpeFile(files[0]))
            self.data_ori = self.spefile_ori.get_raw_spectra_from_spefile()
        else:
            for temp_file in files:
                self.data_ori.append(self._read_txt_or_csv(temp_file))
        return self.data_ori

    def _calibrate(self, remove_baseline=False):
        data = Math.calibrate(self.data_ori, self.data_cal)

        if remove_baseline:
            data, *__ = self.math.baseline_sub(data)

        self.repo.set_data(data)
        return data

    def calibrate(self, remove_baseline=False):
        self._calibrate(remove_baseline)
        self._create_classes()

    def _calibrate_from_ref(self, method, fitting_method='quadratic', laser_wavelength=-1, ref_chemical='cyclo'):
        ref = self.data_cal
        return Math.calibrate_from_ref(ref, method, fitting_method, laser_wavelength, ref_chemical)

    def set(self, **kargs):
        '''Set properties'''
        for key, value in kargs.items():
            self.properties[key] = value
        # self._renew()

    def get(self, *args):
        '''Get properties'''
        ret = []
        for item in args:
            ret.append(self.properties.get(item))
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def clear_settings(self):
        self.properties = {}

    def set_props_from_default(self, *args, force=True):
        # if force==True: set all to default
        # if force==False: add default
        if not args:
            if force:
                self.clear_settings()
                if hasattr(self, 'default_settings'):
                    self.set(**self.default_properties)
            else:
                for key, value in self.default_properties.items():
                    if self.properties.get(key) is None:
                        self.properties[key] = value
        else:
            for item in args:
                try:
                    if force:
                        self.properties[item] = self.default_properties[item]
                    else:
                        for key, value in self.default_properties[item].items():
                            if self.properties[item].get(key) is None:
                                self.properties[item][key] = value
                except:
                    pass

    def retract_background(self, data_input):
        if isinstance(data_input, str):
            if Math.is_file_spe(data_input):
                spe = MySpeFile(data_input)
                data = spe.get_raw_spectra_from_spefile()[0]
            else:
                data = self._read_txt_or_csv(data_input)
        else:
            data = data_input
        self.math.retract(data)

    def export_all(self, silence=False):
        filename = self.properties.get("filenames")[0]
        foldername = os.path.dirname(filename).split("/")[-1]

        if len(self.data) == 1:
            if silence:
                return self.io.export_file(self.data[0], filepath=filename+"_export.csv")
            else:
                return self.io.export_file(self.data[0], default_prefix=foldername)
        else:
            if silence:
                return self.io.export_files(self.data, folder_name=filename+"_export")
            else:
                return self.io.export_files(self.data, prefix=foldername)

    def merge(self, new_ters):
        if new_ters is list:
            new_data = [
                item.data if item is TERS else item for item in new_ters]
            self.repo.set_data(Math.merge_data(self.data, *new_data))

        elif new_ters is TERS:
            self.repo.set_data(Math.merge_data(self.data, new_ters.data))

        elif new_ters is np.ndarray:
            self.repo.set_data(Math.merge_data(self.data, new_ters))
