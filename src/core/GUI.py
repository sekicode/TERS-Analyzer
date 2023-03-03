
import random
import tkinter.filedialog
import tkinter.font as tk_font
import tkinter.messagebox as messagebox
import tkinter.simpledialog
from tkinter import Button, Entry, Frame, Label, StringVar, TclError, ttk
from tkinter.constants import DISABLED, LEFT, NORMAL, RIGHT, TOP

import numpy as np

from src.core.Ask import Ask
from src.utils.error import CancelInterrupt, error_message_box
from src.ters.multi_ters import MultiTERS
from src.utils.setting import Setting
from src.ters.TERS import TERS
from src.core.wrapper import MyWrapper

Version = '3.1.2.1'


class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self._create_variable()
        self._createWidgets()
        self._make_grid()
        self._adjust_grid_spacing()
        self._set_font()
        self.grid()

    def _create_variable(self):
        self.ask = Ask()

    matrix_mapping = data_ori = data_cal = ters = data_retracted = []
    pixel_x = 1
    pady = 5
    color = 'winter'
    result = None

    def _createWidgets(self):
        self.default_settings = {'color': 'winter'}
        self.settings = Setting(file='Settings.json',
                                default_settings=self.default_settings)
        self.data_context = StringVar()
        self.nameInput = Entry(textvariable=self.data_context)
        self.data_context.set("No data")
        '''self.nameInput'''
        self.width_button = 20
        self.height_button = 2

        s = ttk.Style()
        s.configure('TNotebook.Tab', font=(
            'Helvetica', '12'), padding=[12, 5])
        self.myFont_button = tk_font.Font(family='Helvetica', size=12)
        self.myFont_label = tk_font.Font(family='Helvetica', size=15)

        self.label1 = Label(self, text="File")
        self.label2 = Label(self, text="Analyze")
        self.label3 = Label(self, text="Plot")
        self.label4 = Label(self, text="Tools")
        self.label5 = Label(self, text="Settings")

        self.Button11 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               text='Select calibration file'.title(),
                               command=self.click_select_calibration_file)

        self.Button12 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Select raw data'.title(),
                               command=self.click_select_raw_data)

        self.Button13 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               text='Select Calibrated data'.title(),
                               command=self.click_get_calibrated_files)

        self.Button14 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               text='Subtract Background',
                               state=DISABLED,
                               command=self.click_get_retracted_data)

        self.Button15 = Button(self,
                               text='Re-calibrate',
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               command=self.click_calibrate)

        self.Button21 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Smooth', command=self.click_smooth)

        self.Button22 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Cut', command=self.click_cut_data)

        self.Button23 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Remove Baseline', command=self.click_remove_baseline)

        self.Button24 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Remove Spikes', command=self.click_spike_removal)

        self.Button25 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Interpolate Wavenumber', command=self.click_interpolate_wavenumber)

        self.Button26 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Set min zero'.title(), command=self.click_set_min_zero)

        self.Button27 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Delete a Spectrum', command=self.click_delete_spectra)

        self.Button28 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Delete First Spectrum', command=self.click_delete_first_spectra)

        self.Button29 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Merge to Single', command=self.click_merge_to_one)

        self.Button210 = Button(self,
                                width=self.width_button,
                                height=self.height_button,
                                state=DISABLED,
                                text='Normalize', command=self.click_normalizate)

        self.Button31 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED, text='Plot spectrum'.title(), command=self.click_plot_spec)
        self.Button32 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Waterfall mapping'.title(), command=self.click_waterfall_mapping)

        self.Button33 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Point mapping'.title(), command=self.click_point_mapping)

        self.Button34 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='More mappings'.title(), command=self.click_more_mapping)

        self.Button35 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Point lineprofile'.title(), command=self.click_point_lineprofile)

        self.Button36 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='More lineprofiles'.title(), command=self.click_more_lineprofile)

        self.Button41 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Export spectra'.title(),
                               command=self.click_export_file)

        self.Button42 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Batch Convert and Export', command=self.click_batch_convert)

        self.Button51 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               text='Set color'.title(),
                               command=self.click_set_color)

        self.Button52 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Set x Pixels', command=self.click_set_pixel_x)

        self.Button53 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               state=DISABLED,
                               text='Set step size'.title(), command=self.click_set_distance)

        self.Button54 = Button(self,
                               width=self.width_button,
                               height=self.height_button,
                               text='About', command=self.click_show_about)

        self.buttons_disabled = [self.Button26, self.Button27, self.Button29, self.Button14,
                                 self.Button15, self.Button21, self.Button31, self.Button22,
                                 self.Button32, self.Button34, self.Button36,
                                 self.Button33, self.Button41, self.Button23, self.Button24,
                                 self.Button52, self.Button53, self.Button35,
                                 self.Button25, self.Button29, self.Button210, self.Button28, ]

    def _make_grid(self):
        self.grid_pattern = [[self.label1, self.Button11, self.Button12, self.Button13, self.Button14, self.Button15, self.Button41, self.Button42, ],
                             [self.label2, self.Button21, self.Button22, self.Button23, self.Button24, self.Button25,
                                 self.Button26, self.Button27, self.Button28, self.Button29, self.Button210, ],
                             [self.label3, self.Button31, self.Button32, self.Button33,
                                 self.Button34, self.Button35, self.Button36, ],
                             [self.label5, self.Button51, self.Button52, self.Button53, self.Button54, ], ]

        for j, buttons in enumerate(self.grid_pattern):
            for i, button in enumerate(buttons):
                button.grid(row=i, column=j)

    def _set_font(self):
        local_variables = dir(self)
        for item in local_variables:
            obj = getattr(self, item)
            if isinstance(obj, Button):
                obj['font'] = self.myFont_button
            elif isinstance(obj, Label):
                obj['font'] = self.myFont_label

    def _adjust_grid_spacing(self):
        col_count, row_count = self.grid_size()

        for col in range(col_count):
            self.grid_columnconfigure(col, minsize=220)

        for row in range(row_count):
            self.grid_rowconfigure(row, minsize=62)

    def set_window_middle(self, width, height):
        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        w = width
        h = height
        x = int((ws / 2) - (w / 2)-100)
        y = int((hs / 2) - (h / 2)-100)
        self.master.geometry('{}x{}+{}+{}'.format(w, h, x, y))

    @MyWrapper.safe
    def click_select_calibration_file(self):
        flag = False
        file = tkinter.filedialog.askopenfilename()
        if file is None or file == '':
            return None
        self.file_cal = file
        try:
            if file.split('.')[-1] == 'SPE':
                raise TypeError("Cannot read SPE file as a calibration")
        except FileNotFoundError:
            messagebox.showinfo("Error", "Where is the file?")
        except:
            messagebox.showinfo("Error", "Files cannot be read")
        else:
            self._set_all_button_disabled()
            self.Button12.configure(state=NORMAL)
            self.Button42.configure(state=NORMAL)

    @MyWrapper.safe
    def click_select_raw_data(self):
        files = tkinter.filedialog.askopenfilenames()
        if files == '':
            return None
        try:
            self.file_ori = files
            self.ters = TERS(file_cal=self.file_cal,
                             file_ori=files,
                             settings=self.settings,
                             laser_wavelength=getattr(
                                 self, 'laser_wavelength', None),
                             method=getattr(self, 'method', None))
        except FileNotFoundError:
            return None
        except Exception as ex:
            error_message_box(ex, 'Reading files')
        else:
            self.after_calibrate()

    @MyWrapper.safe
    def click_get_calibrated_files(self):
        files = tkinter.filedialog.askopenfilenames()
        if files == '':
            return None
        try:
            self.file_ori = files
            self.ters = TERS(file_cal=None,
                             file_ori=files,
                             settings=self.settings,
                             laser_wavelength=getattr(
                                 self, 'laser_wavelength', None),
                             method=getattr(self, 'method', None))
        except FileNotFoundError:
            return None
        except Exception as ex:
            error_message_box(ex, 'Reading files')
        else:
            self.after_calibrate()
            self.Button12.configure(state=DISABLED)

    @MyWrapper.safe
    def click_get_retracted_data(self):
        file = tkinter.filedialog.askopenfilename()
        if not file:
            return None
        try:
            self.ters.retract_background(file)
        except FileNotFoundError:
            messagebox.showinfo("Error", "Where is the file?")
        except Exception as error:
            error_message_box(error, "Files cannot be read")

    def _set_all_button_available(self):
        for button in self.buttons_disabled:
            button.configure(state=NORMAL)

    def _set_all_button_disabled(self):
        for button in self.buttons_disabled:
            button.configure(state=DISABLED)

    @MyWrapper.safe
    def click_calibrate(self):
        self.ters.calibrate()
        self.after_calibrate()

    def after_calibrate(self):
        self.ters.set(pixel_x=1)
        self.ters.set(distance=0)

        self._set_all_button_available()

    def change_pixel_x(self):
        temp = 1
        lens = len(self.ters)
        for attempt in range(10):
            temp = tkinter.simpledialog.askinteger("Messagebox",
                                                   "The number of x pixel? For a lineprofile, type 1.")
            if temp is None:
                raise CancelInterrupt
            if temp > 0 and lens % temp == 0:
                return temp
            if temp <= 0:
                messagebox.showinfo("Error", "Please enter a positive number.")
            elif lens % temp != 0:
                messagebox.showinfo(
                    "Error", "{} cannot be divided by {}".format(lens, temp))
        raise CancelInterrupt

    @MyWrapper.safe
    def click_set_pixel_x(self):
        self.ters.set(pixel_x=self.change_pixel_x())

    @MyWrapper.safe
    def click_smooth(self, window_size=31, polynomial_order=3):
        for attempt in range(10):
            temp = tkinter.simpledialog.askinteger(
                "Smoothing window size?", "Default = 31")
            if temp is None:
                break
            window_size = temp
            if window_size > polynomial_order:
                try:
                    self.ters.math.smooth(window_size, polynomial_order)
                except:
                    messagebox.showinfo("Error", "Smoothing went wrong.")
                    return 0
                break
            else:
                messagebox.showinfo(
                    "Error", "Please enter a number larger than {}.".format(polynomial_order))

    @MyWrapper.safe
    def click_plot_spec(self):
        return self.ters.plot.plot_spec()

    @MyWrapper.safe
    def click_cut_data(self):
        center, halfwidth = self.ask_for_edges()
        if center is None or halfwidth is None:
            return None
        self.ters.math.cut_data(center, halfwidth)

        # messagebox.showinfo("Error", "Cutting went wrong")
        if messagebox.askyesno("Messagebox", "Subtract background again?"):
            self.click_remove_baseline()

    def _renew_pixel_x(self):
        pixel_x = self.ters.properties.get("pixel_x")
        if not pixel_x or pixel_x == 1:
            self.ters.set(pixel_x=self.change_pixel_x())
        return pixel_x

    @MyWrapper.safe
    def click_point_mapping(self):
        self._renew_pixel_x()
        self.ters.plot.point_mapping(center=self.ask.center())

    @MyWrapper.safe
    def click_point_lineprofile(self):
        self.ters.plot.point_lineprofile(center=self.ask.center())

    @MyWrapper.safe
    def click_waterfall_mapping(self):
        self.ters.plot.waterfall()

    def ask_for_edges(self):
        for attempt in range(10):
            left_edge = tkinter.simpledialog.askfloat(
                "Messagebox", "Left Edge (cm-1)?")
            if left_edge is None:
                return None, None
            if left_edge > 0:
                break
            else:
                messagebox.showinfo("Error", "Please enter a positive integer")

        for attempt in range(10):
            right_edge = tkinter.simpledialog.askfloat(
                "Messagebox", "Right Edge (cm-1)?")
            if right_edge is None:
                return None, None
            elif right_edge >= left_edge:
                center = (left_edge + right_edge) / 2.
                halfwidth = (right_edge - left_edge) / 2.
                return center, halfwidth
            else:
                messagebox.showinfo(
                    "Error", "Right edge cannot be less than left edge")
        return None, None

    def set_distance(self):
        for attempt in range(10):
            temp = tkinter.simpledialog.askfloat(
                "Messagebox", "The distance between each pixel (nm)?")
            if temp is None:
                raise CancelInterrupt
            elif temp > 0:
                return temp
            else:
                messagebox.showinfo("Error", "Please enter a positive integer")
        raise CancelInterrupt

    @MyWrapper.safe
    def click_set_distance(self):
        self.ters.set(distance=self.set_distance())

    def ask_and_do_method(self, text, lists, default, func):
        popwindow = tkinter.Toplevel(self)
        lab = ttk.Label(popwindow,
                        text=text,
                        font=8)
        lab.pack(side=TOP)

        cb = ttk.Combobox(popwindow,
                          font=8)
        cb['value'] = lists
        cb.set(default)
        cb.pack(side=LEFT)

        bn = ttk.Button(popwindow,
                        text='Okay',
                        command=lambda: finish_ask(func, cb.get(), lists, popwindow, self.ters))
        bn.pack(side=RIGHT,
                pady=self.pady)

        def finish_ask(func, cb_value, lists, window, xyzdata):
            if cb_value not in lists:
                messagebox.showinfo("Error", "This method cannot be found.")
                return None
            try:
                self.result = func(xyzdata, method=cb_value)
            except Exception as ex:
                error_message_box(ex, func.__name__)
            else:
                if func.__name__ == "baseline_sub" or func.__name__ == "change_color":
                    try:
                        window.destroy()
                    except TclError:
                        pass

    @MyWrapper.safe
    def click_remove_baseline(self):
        self.ters.math.baseline_sub()

    @MyWrapper.safe
    def click_set_color(self):
        text = 'Select mapping color'
        list = ('viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples',
                'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
                'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary',
                'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer',
                'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat',
                'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'twilight',
                'twilight_shifted', 'hsv', 'Pastel1', 'Pastel2', 'Paired', 'Accent',
                'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c',
                'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot',
                'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow',
                'jet', 'turbo', 'nipy_spectral', 'gist_ncar')
        default = self.settings.get('color')
        func = self.change_color
        self.ask_and_do_method(text, list, default, func)

    def change_color(self, xyzdata, method):
        self.settings.set(color=method)

    def set_mapping(self, xyzdata, method):
        if method == 'Color':
            self.click_set_color()
        elif method == 'X pixels':
            self.change_pixel_x()
        elif method == 'Distance between each pixel':
            self.set_distance()

    @MyWrapper.safe
    def click_more_mapping(self):
        self._renew_pixel_x()
        center, halfwidth = self.ask_for_edges()
        self.ters.plot.more_mapping(center, halfwidth)

    @MyWrapper.safe
    def click_more_lineprofile(self):
        center, halfwidth = self.ask_for_edges()

        self.ters.plot.more_lineprofile(center, halfwidth)

    @MyWrapper.safe
    def click_show_about(self):
        messagebox.showinfo(
            "About", "Version: {}.\nMade by Zhongyi Lu.".format(Version))

    @MyWrapper.safe
    def click_spike_removal(self):
        info_spike = self.ters.math.find_spike()
        if info_spike is None or info_spike == []:
            messagebox.showinfo("Finding spikes", "No spikes found.")
        else:
            message = ''
            for temp in info_spike:
                message += 'Index of spectrum: {:d}\nWavenumber: {:.2f} cm-1\n\n'.format(
                    temp[0], temp[1])
            if len(info_spike) <= 1:
                message += "Do you want to remove this spike?"
            else:
                message += "Do you want to remove these spikes?"
            if messagebox.askyesno("Finding spikes", message):
                self.ters.math.reduce_spike(info_spike)

    @MyWrapper.safe
    def click_interpolate_wavenumber(self):
        scaling = tkinter.simpledialog.askfloat("Messagebox", "Scaling?")
        if scaling is None:
            return None
        elif scaling > 0:
            self.ters.math.interpolate(scaling)
            return scaling
        else:
            messagebox.showinfo("Error", "Please enter a positive integer")

    def ask_for_positive_float(self, message):
        for i in range(10):
            temp = tkinter.simpledialog.askfloat("Messagebox", message)
            if temp is None:
                return None
            elif temp > 0:
                return temp
            else:
                messagebox.showinfo("Error", "Please enter a positive integer")

    @MyWrapper.safe
    def click_export_file(self):
        return self.ters.export_all()

    @MyWrapper.safe
    def click_set_min_zero(self):
        return self.ters.math.set_min_zero()

    @MyWrapper.safe
    def click_delete_spectra(self):
        index = self.ask.positive_int("Index?")
        return self.ters.math.delete_spectra(index - 1)

    @MyWrapper.safe
    def click_delete_first_spectra(self):
        index = 1
        return self.ters.math.delete_spectra(index - 1)

    @MyWrapper.safe
    def click_batch_convert(self):
        files = tkinter.filedialog.askopenfilenames()
        if files == '':
            return None
        multi_ters = MultiTERS(file_cal=self.file_cal,
                               file_ori=files,
                               settings=self.settings,
                               laser_wavelength=getattr(
                                   self, 'laser_wavelength', None),
                               method=getattr(self, 'method', None))
        multi_ters.export_all_no_options()
        messagebox.showinfo(title="Message Box", message="Batch convert done.")

    @MyWrapper.safe
    def click_merge_to_one(self):
        return self.ters.math.merge_to_one()

    @MyWrapper.safe
    def click_normalizate(self):
        return self.ters.math.normalization()
