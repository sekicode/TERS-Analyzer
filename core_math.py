import copy
import os

import numpy as np
import scipy as sp
from BaselineRemoval import BaselineRemoval
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

from core_cluster import MyCluster
from core_repository import Repository
from core_winspec import SpeFile


class NotFinishedYet(Exception):
    pass


class Math():
    def __init__(self) -> None:
        pass

    def get_mapping_axes(distance, pixel_x, pixel_y, export=False):
        if distance:
            if export:
                x = np.linspace(0, distance * (pixel_x-1), pixel_x)
                y = np.linspace(distance * (pixel_y-1), 0, pixel_y)
                shading = None
            else:
                x = np.linspace(0, distance * pixel_x, pixel_x+1)
                y = np.linspace(distance * pixel_y, 0, pixel_y+1)
                shading = 'flat'
        else:
            x = np.linspace(1, pixel_x, pixel_x)
            y = np.linspace(pixel_y, 1, pixel_y)
            shading = 'nearest'

        return x, y, shading

    def calibrate(ori, cal=None):
        result = np.array(ori, dtype=float)

        # Calibrate spectra if cal is not None
        if cal is not None:
            cal = np.array(cal)
            assert np.shape(result[0]) == np.shape(cal)

            for i in range(len(result)):
                result[i, :, 0] = cal[:, 0]

        for i, spectrum in enumerate(result):
            if (spectrum[:, 0] != result[0, :, 0]).any():
                raise TypeError(
                    "Spectrum {} has different x axis.".format(i+1))

        return result

    def _baseline_sub(xydata: np.ndarray, method="ZhangFit"):
        baseline = copy.deepcopy(xydata)
        baseObj = BaselineRemoval(xydata[:, 1])
        if method == "ZhangFit":
            xydata[:, 1] = baseObj.ZhangFit()
        elif method == "ModPoly":
            xydata[:, 1] = baseObj.ModPoly()
        else:
            xydata[:, 1] = baseObj.IModPoly()
        baseline[:, 1] -= xydata[:, 1]
        return xydata, baseline

    def baseline_sub(data: np.ndarray, method="ZhangFit"):
        data = copy.deepcopy(data)
        baseline = copy.deepcopy(data)

        dim = data.ndim

        if dim == 3:
            for i, xydata in enumerate(data):
                xydata, baseline[i] = Math._baseline_sub(xydata)
        else:
            data, baseline = Math._baseline_sub(data)

        return data, baseline

    def set_min_zero(data_input, center=None, halfwidth=None):
        output = np.array(data_input)
        if center is not None and halfwidth is not None:
            output = Math.cut_data(output, center, halfwidth)

        dim = output.ndim
        if dim == 1:
            output -= np.amin(output)
        elif dim == 2:
            output[:, 1] -= np.amin(output[:, 1])
        elif dim == 3:
            output[:, :, 1] -= np.amin(output[:, :, 1])
        else:
            raise TypeError

        return output

    def smooth(data, window_size=11, polynomial_order=3):
        data = copy.deepcopy(data)
        window_size = int(window_size / 2) * 2 + 1
        polynomial_order = int(polynomial_order)
        dim = len(np.shape(data))
        if dim == 3:
            for i, v in enumerate(data):
                data[i, :, 1] = savgol_filter(
                    data[i, :, 1], window_size, polynomial_order)
        elif dim == 2:
            data[:, 1] = savgol_filter(
                data[:, 1], window_size, polynomial_order)
        elif dim == 1:
            data = savgol_filter(data, window_size, polynomial_order)
        else:
            raise TypeError
        return data

    def interpolate_1d(xdata, scaling=10.):
        lens = len(xdata)
        pointnumber = int(len(xdata) * scaling)
        xold = np.linspace(1, lens, lens)
        xnew = np.linspace(1, lens, pointnumber)
        f = sp.interpolate.interp1d(xold, xdata, kind="quadratic")
        ynew = f(xnew)
        return ynew

    def interpolate_2d(xydata, scaling=10.):
        pointnumber = int(len(xydata[:, 0]) * scaling)
        xnew = np.linspace(xydata[0, 0], xydata[-1, 0], pointnumber)
        f = sp.interpolate.interp1d(
            xydata[:, 0], xydata[:, 1], kind="quadratic")
        ynew = f(xnew)
        data_output = np.transpose(np.vstack((xnew, ynew)))
        return data_output

    def interpolate(xyzdata: np.ndarray, scaling=10):
        shape = list(np.shape(xyzdata))
        shape[1] = int(shape[1]*scaling)
        output = np.zeros(shape=shape)
        for i, v in enumerate(xyzdata):
            output[i] = Math.interpolate_2d(v, scaling)
        return output

    def interpolate_mapping(matrix, scaling_x, scaling_y=-1):
        if scaling_y == -1:
            scaling_y = scaling_x
        rows, cols = np.shape(matrix)
        x = np.linspace(1, cols, cols)
        y = np.linspace(1, rows, rows)
        f = sp.interpolate.interp2d(x, y, matrix, kind="cubic")

        xnew = np.linspace(1, cols, int(cols * scaling_x))
        ynew = np.linspace(1, rows, int(rows * scaling_y))
        return f(xnew, ynew)

    def reorganize_data(xyzdata_input, pixel_x):
        xyzdata = copy.deepcopy(xyzdata_input)
        pixel_x = int(pixel_x)
        lens = len(xyzdata)
        if lens % pixel_x != 0:
            return None
        pixel_y = int(len(xyzdata) / pixel_x)

        new = np.zeros(shape=(len(xyzdata[0, :, 0]), pixel_x, pixel_y))
        for i in range(lens):
            x = i % pixel_x
            y = int(i / pixel_x)
            new[:, x, y] = xyzdata[i, :, 1]
        return new

    def reorganize_back_data(xyzdata_input, new):
        xyzdata = copy.deepcopy(xyzdata_input)
        result = []
        lens, pixel_x, pixel_y = np.shape(new)
        col1 = xyzdata[0, :, 0]
        for i in range(pixel_x * pixel_y):
            x = i % pixel_x
            y = int(i / pixel_x)
            if x == 16:
                a = 1
            col2 = new[:, x, y]
            result.append(np.vstack((col1, col2)).T)
        return result

    def find_point(xydata: np.ndarray, center):
        '''Get index from certain value'''
        if not isinstance(xydata, np.ndarray):
            xydata = np.array(xydata)
        i = 0
        dim = xydata.ndim
        if dim == 2:
            data = xydata[:, 0]
        elif dim == 1:
            data = xydata
        else:
            raise TypeError(
                "Find_point can only accept data no more than 2 dimensions.")
        lens = len(data)
        while i < lens - 1 and data[i] < center:
            i += 1
        if i > 0 and abs(data[i-1] - center) < abs(data[i] - center):
            i -= 1
        return i

    def find_edge(xdata, center, halfwidth):
        '''Get left and right indexes from center value and halfwidth'''
        if halfwidth < 0:
            return None, None
        left_edge = Math.find_point(xdata, center-halfwidth)
        right_edge = Math.find_point(xdata, center+halfwidth)
        return left_edge, right_edge

    def find_center(left_val, right_val):
        '''Get center value and halfwidth from left and right values'''
        center = (left_val+right_val) / 2
        halfwidth = abs(left_val-right_val) / 2
        return center, halfwidth

    def wavelength_to_wavenumber(wavelength1, wavelength2=-1):
        if wavelength2 == -1:
            return 1E7/wavelength1
        else:
            return 1E7/wavelength1 - 1E7/wavelength2

    def wavenumber_to_wavelength(wavenumber1, wavenumber2=0):
        return 1E7/(wavenumber1-wavenumber2)

    def wavelength_to_wavelength(wavelength1, wavelength2=-1):
        Math.wavenumber_to_wavelength(
            Math.wavelength_to_wavenumber(wavelength1, wavelength2))

    def integrate_spec(xydata, center, halfwidth):
        if halfwidth < 0:
            return None
        # interpolated_data = interpolate_2d([xydata][0], 50000)
        # data = cut_data(interpolated_data, center, halfwidth)
        data = Math.cut_data(xydata, center, halfwidth)
        x = data[:, 0]
        y = data[:, 1]
        return integrate.trapz(y, x)

    def pick_peaks(xydata, center=-1, halfwidth=-1, prominence=200):
        xydata = copy.deepcopy(xydata)
        # if something wrong, transpose input matrix
        if center == -1 and halfwidth == -1:
            data = xydata
        else:
            (left_edge, right_edge) = Math.find_edge(
                xydata[:, 0], center, halfwidth)
            data = xydata[left_edge:right_edge, :]

        data, baseline = Math.baseline_sub(data)
        data = Math.normalization(data)
        peaks_info = find_peaks(data[:, 1], prominence=prominence)

        peaks_index = peaks_info[0]
        peaks = data[peaks_index, :]

        peak_prom = peaks_info[1]['prominences']
        peak_prom = np.array(peak_prom).reshape(1, len(peak_prom))
        return np.c_[peaks, peak_prom.T]

    def get_highest_peaks(xydata, num=-1, center=-1, halfwidth=-1, prominence=200):
        # if something wrong, transpose input matrix
        peaks = Math.pick_peaks(xydata, center, halfwidth, prominence)
        sorted_peaks_index = peaks[:, 2].argsort()
        if num != -1 or num < len(sorted_peaks_index):
            sorted_peaks_index = sorted_peaks_index[-1:-1 - num:-1]

        return peaks[sorted_peaks_index]

    def report_highest_peaks(xydata, **kargs):
        result = ""
        peaks = Math.pick_peaks(xydata, **kargs)
        for peak in peaks:
            result += "{:.1f} cm-1\tprominence={:.0f}\n".format(
                peak[0], peak[2])
        # print(result)
        return result

    def cut_data(data_input: np.ndarray, center, halfwidth):
        data = copy.deepcopy(data_input)
        dim = data.ndim
        if dim == 3:
            for xydata in data:
                '''这个要想办法处理'''
                if (xydata[:, 0] != data[0, :, 0]).any():
                    return None

            (left_edge, right_edge) = Math.find_edge(
                data[0, :, 0], center, halfwidth)
            return data[:, left_edge:right_edge]

        elif dim == 1:
            (left_edge, right_edge) = Math.find_edge(data, center, halfwidth)
            return data[left_edge:right_edge]

        elif dim == 2:
            (left_edge, right_edge) = Math.find_edge(
                data[:, 0], center, halfwidth)
            return data[left_edge:right_edge, :]

        else:
            raise TypeError

    def reshape_mapping_data(value_mapping, pixel_x):
        len_value_mapping = len(value_mapping)
        pixel_y = int(len_value_mapping / pixel_x)
        return np.array(value_mapping).reshape(pixel_y, pixel_x)

    def find_highest_index(xdata):
        temp = xdata[0]
        highest_index = 0
        for i in range(len(xdata)):
            if xdata[i] > temp:
                highest_index = i
                temp = xdata[i]
        return highest_index

    def get_waterfall_data(xyzdata: np.ndarray):
        size_lineprofile = len(xyzdata)
        size_x_axis = len(xyzdata[0, :, 0])
        data_lineprofile = np.zeros(shape=(size_lineprofile, size_x_axis))
        # for i in range(size_lineprofile):
        #     data_lineprofile[i, :] = xyzdata[size_lineprofile - i - 1, :, 1]
        data_lineprofile = xyzdata[::-1, :, 1]
        return data_lineprofile

    def get_waterfall_xyz(xyzdata: np.ndarray, distance, export=False):
        z = Math.get_waterfall_data(xyzdata)
        x = xyzdata[0, :, 0]
        lens = len(z)
        if distance:
            if export:
                y = np.linspace((lens-1) * distance, 0, lens)
            else:
                y = np.linspace((lens-0.5) * distance, distance*0.5, lens)
        else:
            y = np.arange(lens, 0, -1)

        return x, y, z

    def sheet_export(x, y, z):
        shape = z.shape
        output = np.zeros(shape=(shape[0]+1, shape[1]+1))
        output[1:, 0] = y
        output[0, 1:] = x
        output[1:, 1:] = z
        return output

    def get_waterfall_export(xyzdata: np.ndarray, distance):
        x, y, z = Math.get_waterfall_xyz(xyzdata, distance, export=True)
        if z is None:
            return None

        return Math.sheet_export(x, y, z)

    def get_integration_mapping_data(xyzdata, center, halfwidth, pixel_x, method):
        if method == 'Gaussian Fit':
            method = 'normal'
        elif method == 'Peak finding & Gaussian Fit':
            method = 'smart'

        lens = len(xyzdata)
        value_mapping = np.zeros(shape=lens)

        if halfwidth and halfwidth > 0:
            if method == 'normal':
                for i in range(lens):
                    value_mapping[i] = Math.integrate_spec(
                        xyzdata[i], center, halfwidth)

            elif method == 'smart':
                for i in range(lens):
                    data = Math.cut_data(xyzdata[i], center, halfwidth)
                    highest_index = Math.find_highest_index(data[:, 1])
                    highest_peak_wavenumber = data[highest_index, 0]
                    value_mapping[i] = Math.integrate_spec(
                        xyzdata[i], highest_peak_wavenumber, 10)

        return np.flipud(Math.reshape_mapping_data(value_mapping, pixel_x))

    def get_point_mapping_data(xyzdata, center, pixel_x=1):
        lens = len(xyzdata)
        value_mapping = np.zeros(shape=lens)
        for i in range(lens):
            value_mapping[i] = xyzdata[i, Math.find_point(
                xyzdata[i], center), 1]

        return np.flipud(Math.reshape_mapping_data(value_mapping, pixel_x))

    def get_point_mapping_xyz(xyzdata: np.ndarray, center: float, pixel_x: int, distance: float, export=False):
        z = Math.get_point_mapping_data(xyzdata, center, pixel_x)
        lens = len(xyzdata)
        pixel_y = int(lens / pixel_x)

        x, y, shading = Math.get_mapping_axes(
            distance, pixel_x, pixel_y, export)

        return x, y, z, shading

    def get_point_mapping_export(xyzdata: np.ndarray, center: float, pixel_x: int, distance: float):
        x, y, z, _ = Math.get_point_mapping_xyz(
            xyzdata, center, pixel_x, distance, export=True)
        return Math.sheet_export(x, y, z)

    def get_point_mapping_data_inter(xyzdata, center, pixel_x=1):
        lens = len(xyzdata)
        value_mapping = np.zeros(shape=lens)
        for i in range(lens):
            value_mapping[i] = xyzdata[i, Math.find_point(
                xyzdata[i], center), 1]

        reshaped_mapping = np.flipud(
            Math.reshape_mapping_data(value_mapping, pixel_x))
        reshaped_mapping = Math.interpolate_mapping(reshaped_mapping, 10)
        return reshaped_mapping

    def gaussian(x, *param):
        y = param[2] * np.exp(-(x - param[0]) ** 2 / (2 * param[1] ** 2))
        return y

    def guassian_fit(xydata, center, halfwidth):
        x = Math.cut_data(xydata, center, halfwidth)[:, 0]
        y = Math.set_min_zero(Math.cut_data(xydata, center, halfwidth)[:, 1])
        popt, pcov = curve_fit(Math.gaussian, x, y, p0=[center, 6, 100])
        popt[1] = abs(popt[1] * 2.355)
        return popt

    def linearfit(xold, yold, xnew):
        poly = np.polyfit(xold, yold, 1)
        a = np.array([xnew, 1]).T
        return np.dot(poly, a)

    def polyfit_2(xold, yold, xnew):
        poly = np.polyfit(xold, yold, 2)
        a = np.array([xnew * xnew, xnew, 1]).T
        return np.dot(poly, a)

    def polyfit_3(xold, yold, xnew):
        poly = np.polyfit(xold, yold, 3)
        a = np.array([xnew * xnew * xnew, xnew * xnew, xnew, 1]).T
        return np.dot(poly, a)

    def get_shift_mapping_data(xyzdata, center, halfwidth=10.0, pixel_x=1, method='normal'):
        if method == 'Normal':
            method = 'normal'
        elif method == 'Gaussian Fit':
            method = 'smart'
        elif method == 'Peak finding & Gaussian Fit':
            method = 'super smart'

        size_z = len(xyzdata)
        value_mapping = np.empty(size_z)
        if method == 'normal':
            for i in range(size_z):
                temp = xyzdata[i]
                data = Math.cut_data(temp, center, halfwidth)
                highest_index = Math.find_highest_index(data[:, 1])
                value_mapping[i] = data[highest_index, 0]

        elif method == 'smart':
            # elif method == "super smart":
            for i in range(size_z):
                temp = xyzdata[i]
                data = Math.cut_data(temp, center, halfwidth)
                highest_index = Math.find_highest_index(data[:, 1])
                highest_center = data[highest_index, 0]
                value_mapping[i] = Math.guassian_fit(
                    temp, highest_center, halfwidth=10)[0]

        median_peak_position = np.median(value_mapping)
        value_mapping -= median_peak_position
        return np.flipud(Math.reshape_mapping_data(value_mapping, pixel_x)), median_peak_position

    def get_sigma_mapping_data(xyzdata, center, halfwidth=10.0, pixel_x=1, method='normal'):
        if method == 'Gaussian Fit':
            method = 'normal'
        elif method == 'Peak finding & Gaussian Fit':
            method = 'smart'

        lens = len(xyzdata)
        value_mapping = np.empty(shape=lens)
        if method == 'normal':
            for i in range(lens):
                value_mapping[i] = Math.guassian_fit(
                    xyzdata[i], center, halfwidth)[1]

        elif method == 'smart':
            for i in range(lens):
                data = Math.cut_data(xyzdata[i], center, halfwidth)
                highest_index = Math.find_highest_index(data[:, 1])
                highest_center = data[highest_index, 0]
                value_mapping[i] = Math.guassian_fit(
                    xyzdata[i], highest_center, halfwidth=10)[1]

        return np.flipud(Math.reshape_mapping_data(value_mapping, pixel_x))

    def get_point_lineprofile_xy(xyzdata, center, distance):
        lens = len(xyzdata)
        matrix_mapping = Math.get_point_mapping_data(
            xyzdata, center, pixel_x=1)

        if distance:
            x = np.arange(0, lens*distance, distance)[::-1]
        else:
            x = np.arange(1, lens+1, 1)[::-1]

        y = matrix_mapping[:, 0]
        return x, y

    def get_point_lineprofile_export(xyzdata, center, distance):
        xdata, ydata = Math.get_point_lineprofile_xy(
            xyzdata, center, distance)
        return np.c_[xdata, ydata]

    def get_time_from_spefile(spefile):
        ExperimentTimeLocal = int(spefile.header.ExperimentTimeLocal)
        hour = int(ExperimentTimeLocal / 10000)
        minute = int((ExperimentTimeLocal % 10000) / 100)
        second = int(ExperimentTimeLocal % 100)
        return '{:d}:{:0>2d}:{:0>2d}'.format(hour, minute, second)

    def get_timezone_from_spefile(spefile):
        time_zone = int((int(spefile.header.ExperimentTimeLocal) -
                         int(spefile.header.ExperimentTimeUTC)) / 10000)
        if time_zone < -12:
            time_zone += 24
        elif time_zone > 12:
            time_zone -= 24
        return time_zone

    def get_info_from_spefile(spefile):
        adc = spefile.adc
        adc_rate = spefile.adc_rate
        time = Math.get_time_from_spefile(spefile)
        time_zone = Math.get_timezone_from_spefile(spefile)
        frame = np.shape(spefile.data)[0]
        return adc, adc_rate, frame, time, time_zone

    def read_winspec_file(spefile):
        Math.get_info_from_spefile(spefile)
        return None

    def get_raw_spectra_from_spefile(spefile: SpeFile):
        '''Get only intensity from spefile,
        wavenumbers are all set to zero and wait for later calibration'''
        data = np.array(spefile.data)
        shape = data.shape
        result = np.empty(shape=(shape[0], shape[1], 2))
        # for item in result:
        #     item[:, 0] = np.linspace(1, shape[1], shape[1])
        result[:, :, 0] = spefile.xaxis
        result[:, :, 1] = data[:, :, 0]
        return result

    def calibrate_from_ref(ref, method, fitting_method='quadratic', laser_wavelength=-1, ref_chemical='cyclo'):
        '''Overwrite wavenumber columns by a fitting from reference peaks
        Caution: Input will be changed!'''
        ref_inter = ref
        # ref_inter = interpolate_2d(ref, 100)

        peaks_ref = np.sort(Math.get_highest_peaks(ref_inter, 4)[:, 0])
        if ref_chemical == 'cyclo':
            REFERENCE_PEAKS = [801.3, 1028.3, 1266.4, 1444.4]
        peaks_liter = np.array(REFERENCE_PEAKS)
        if method == "cm-1":
            peaks_liter = 1E7 / peaks_liter
        elif method == "ref cm-1":
            peaks_liter = 1E7/(1E7/laser_wavelength-peaks_liter)
        elif method == "nm":
            None

        # peaks_ref = np.array([388., 569., 768., 920.])

        FIT_METHOD = {'linear': Math.linearfit,
                      'quadratic': Math.polyfit_2,
                      'cubic': Math.polyfit_3}

        ref[:, 0] = FIT_METHOD[fitting_method](
            peaks_ref, peaks_liter, ref[:, 0])

        if method == "cm-1":
            ref[:, 0] = 1E7 / ref[:, 0]
        elif method == "ref cm-1":
            ref[:, 0] = 1E7 / laser_wavelength - 1E7 / ref[:, 0]
        elif method == "nm":
            None

        # export_spectrum(ref)
        return ref

    def find_spike(xyzdata: np.ndarray, Fit_halfwidth=5, FWHM_threshold=3):
        xyzdata = copy.deepcopy(xyzdata)
        info_spike = []
        data_without_baseline = Math.baseline_sub(xyzdata)[0]

        for index, temp in enumerate(data_without_baseline):
            spike_wavenumber = temp[np.argmax(temp[:, 1]), 0]
            if spike_wavenumber - Fit_halfwidth > temp[0, 0] and spike_wavenumber + Fit_halfwidth < temp[-1, 0]:
                try:
                    FWHM = Math.guassian_fit(
                        temp, spike_wavenumber, Fit_halfwidth)[1]
                    if FWHM <= FWHM_threshold:
                        info_spike.append([index, spike_wavenumber])
                except:
                    pass

        return info_spike

    def reduce_spike(xyzdata: np.ndarray, info_spike=None, Fit_halfwidth=5, FWHM_threshold=3, Removal_halfwidth=3):
        xyzdata = copy.deepcopy(xyzdata)
        if info_spike is None:
            info_spike = Math.find_spike(
                xyzdata, Fit_halfwidth, FWHM_threshold)

        baseline = Math.baseline_sub(xyzdata)[1]
        for temp in info_spike:
            index_spec = temp[0]
            wavenumber = temp[1]
            halfwidth = Removal_halfwidth
            index_peak = Math.find_point(xyzdata[index_spec], wavenumber)
            xyzdata[index_spec, index_peak-halfwidth:index_peak+halfwidth, 1] \
                = baseline[index_spec, index_peak-halfwidth:index_peak+halfwidth, 1]
        return xyzdata

    def retract(xyzdata, data_sub):
        xyzdata = copy.deepcopy(xyzdata)
        for xydata in xyzdata:
            xydata[:, 1] -= data_sub[:, 1]
        return xyzdata

    def copy_list(data_old: list, data_new: list):
        if id(data_old) == id(data_new):
            return None
        data_old.clear()
        for item in data_new:
            data_old.append(item)

    def get_cluster_mapping_data(data, pixel_x, n_components=10, n_clusters=3):
        data = copy.deepcopy(data)
        cluster = MyCluster()
        pca_result = cluster.do_pca(data, n_components)
        value_mapping = cluster.kmeans(pca_result, n_clusters)

        return np.flipud(Math.reshape_mapping_data(value_mapping, pixel_x))

    def get_pca_variance_ratio(data, n_components=10):
        data = copy.deepcopy(data)
        cluster = MyCluster()
        cluster.do_pca(n_components)
        return cluster.pca_variance_ratio_

    def delete_spectra(data, index):
        data = copy.deepcopy(data)
        return np.delete(data, index, axis=0)

    def merge_data_single_file(old_data, new_data):
        if new_data is not np.ndarray:
            new_data = np.array(new_data)
        assert old_data.shape[1:] == new_data.shape[1:]
        return np.concatenate(old_data, new_data, axis=0)

    def merge_data_list(old_data, new_data):
        for item in new_data:
            if item.shape[1:] != old_data.shape[1:]:
                raise TypeError
        return np.concatenate(old_data, *new_data, axis=0)

    def merge_data(old_data, new_data):
        old_data = copy.deepcopy(old_data)
        new_data = copy.deepcopy(new_data)

        if new_data is list and new_data[0] is np.ndarray and new_data[0].ndim == 3:
            return Math.merge_spe_list(old_data, new_data)
        else:
            return Math.merge_spe_single_file(old_data, new_data)

    def merge_to_one(data: np.ndarray):
        first_spectrum = data[0]
        for i, v in enumerate(data):
            if not (v[:, 0] == first_spectrum[:, 0]).all():
                raise TypeError("Spectrum {} does not match.".format(i+1))
        shape = data.shape
        result = np.empty(shape=(1, shape[1], shape[2]))
        result[0, :, 0] = data[0, :, 0]
        result[0, :, 1] = data[:, :, 1].sum(axis=0)
        return result

    def _normalize_2d(data: np.ndarray, scale=1000):
        max = np.max(data[:, 1])
        min = np.min(data[:, 1])
        _range = max - min
        data[:, 1] = (data[:, 1] - min) * scale / _range
        return data

    def normalization(data: np.ndarray, scale=1000):
        data = np.array(data)
        if data.ndim == 3:
            for spectrum in data:
                spectrum = Math._normalize_2d(spectrum, scale)
        elif data.ndim == 2:
            data = Math._normalize_2d(data, scale)
        return data

    def get_extension_from_filename(filename: str):
        filepath, file_extension = os.path.splitext(filename)
        return file_extension

    def is_file_spe(filename: str):
        if filename is None or filename == '':
            return None
        elif filename.split('.')[-1] == 'SPE':
            return True
        else:
            return False


class TERS_Math():
    def __init__(self, repo: Repository, properties: dict) -> None:
        # self.repo.data = data
        self.properties = properties
        self.cluster = MyCluster()
        self.repo = repo

    @property
    def data(self):
        return self.repo.data

    def decorator(func):
        def wrapper(self, *args, **kargs):
            data = func(self, *args, **kargs)
            inline = kargs.get('inline')
            if inline is None or inline:
                # Math.copy_list(self.repo.data, data)
                self.repo.set_data(data)
            else:
                return data
        return wrapper

    def baseline_sub(self, method="ZhangFit", inline=True):
        data, baseline = Math.baseline_sub(self.repo.data, method)
        if inline:
            # Math.copy_list(self.repo.data, data)
            self.repo.set_data(data)
        else:
            return data, baseline

    @decorator
    def smooth(self, window_size=11, polynomial_order=3, inline=True):
        return Math.smooth(self.repo.data, window_size, polynomial_order)

    @decorator
    def interpolate(self, scaling=10, inline=True):
        return Math.interpolate(self.repo.data, scaling)

    @decorator
    def cut_data(self, center, halfwidth, inline=True):
        return Math.cut_data(self.repo.data, center, halfwidth)

    def get_waterfall_data(self):
        return Math.get_waterfall_data(self.repo.data)

    def get_integration_mapping_data(self, center, halfwidth=3.0, pixel_x=1, method='normal'):
        return Math.get_integration_mapping_data(self.repo.data, center, halfwidth, pixel_x, method)

    def get_point_mapping_data(self, center, pixel_x=1):
        return Math.get_point_mapping_data(self.repo.data, center, pixel_x)

    def get_point_mapping_data_inter(self, center, pixel_x=1):
        return Math.get_point_mapping_data_inter(self.repo.data, center, pixel_x)

    def get_shift_mapping_data(self, center, halfwidth=10.0, pixel_x=1, method='normal'):
        return Math.get_shift_mapping_data(self.repo.data, center, halfwidth, pixel_x, method)

    def get_sigma_mapping_data(self, center, halfwidth=10.0, pixel_x=1, method='normal'):
        return Math.get_sigma_mapping_data(self.repo.data, center, halfwidth, pixel_x, method)

    def find_spike(self, Fit_halfwidth=5, FWHM_threshold=3):
        return Math.find_spike(self.repo.data, Fit_halfwidth, FWHM_threshold)

    @decorator
    def reduce_spike(self, info_spike=None, Fit_halfwidth=5, FWHM_threshold=3, Removal_halfwidth=3, inline=True):
        return Math.reduce_spike(self.repo.data, info_spike, Fit_halfwidth, FWHM_threshold, Removal_halfwidth)

    @decorator
    def retract(self, data_sub, inline=True):
        return Math.retract(self.repo.data, data_sub)

    @decorator
    def set_min_zero(self, inline=True):
        return Math.set_min_zero(self.data)

    @decorator
    def delete_spectra(self, index, inline=True):
        return Math.delete_spectra(self.data, index)

    @decorator
    def merge_to_one(self, inline=True):
        return Math.merge_to_one(self.data)

    @decorator
    def normalization(self, inline=True):
        return Math.normalization(self.data)
