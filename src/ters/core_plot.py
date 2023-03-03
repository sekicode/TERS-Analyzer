import copy
import os
import time
import tkinter.filedialog
import tkinter.simpledialog

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button as pltButton
from matplotlib.widgets import RadioButtons, Slider, TextBox

from src.core.core_Ask import Ask
from src.utils.core_error import CancelInterrupt, error_message_box
from src.utils.core_IO import TERS_IO
from src.ters.core_math import Math, TERS_Math
from src.ters.core_plot_arrange import PlotArrange
from src.utils.core_repository import Repository
from src.utils.core_setting import Setting

matplotlib.use("TkAgg")


class TERS_Plot():
    def __init__(self, repo: Repository, math: TERS_Math, io: TERS_IO, properties: dict, setttings: Setting) -> None:
        self.io = io
        self.arrange = PlotArrange()
        self.repo = repo
        self.properties = properties
        self.settings = setttings

    def _get(self, name, default, **kargs):
        for dic in [kargs, self.properties, self.settings]:
            if dic.get(name) is not None:
                return dic.get(name)
        return default

    def _get_pixel_x(self, **kargs):
        return self._get(name='pixel_x', default=1, **kargs)

    def _get_color(self, **kargs):
        return self._get(name='color', default='winter', **kargs)

    def _get_distance(self, **kargs):
        return self._get(name='distance', default=1, **kargs)

    def _get_data(self):
        return np.array(self.repo.data)

    def _get_filepaths(self, **kargs):
        return self._get(name='filenames', default='Spectra', **kargs)

    def _get_prefix(self, **kargs):
        filename = self._get_filepaths(**kargs)[0]
        return os.path.dirname(filename).split("/")[-1]

    def _get_windows_title(self, **kargs):
        filepaths = self._get_filepaths()
        if len(filepaths) == 1:
            path, name = os.path.split(filepaths[0])
        else:
            name = filepaths[0].split('/')[-2]
        return name

    def _set_slider(self, val, slider):
        if val < slider.valmin:
            val = slider.valmin
        elif val > slider.valmax:
            val = slider.valmax
        slider.set_val(val)

    def _sub_slider(self, slider, step=1):
        new_val = int(slider.val - step)
        self._set_slider(new_val, slider)

    def _add_slider(self, slider, step=1):
        new_val = int(slider.val + step)
        self._set_slider(new_val, slider)

    def _set_xaxis_int(self, ax):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _set_yaxis_int(self, ax):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def _check_center(self, center):
        if center is None:
            data = self._get_data()
            center = np.median(data[0][:, 0])
        return center

    def _check_halfwidth(self, halfwidth):
        return halfwidth if halfwidth and halfwidth > 0 else 10

    def plot_spec(self, index=0, **kargs):
        xyzdata = self._get_data()
        windows_title = self._get_windows_title()
        if index < 0 and index >= len(xyzdata):
            return None

        lens = len(xyzdata)
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(windows_title)
        self.arrange.adjust_figure_position()
        xdata = xyzdata[index, :, 0]
        ydata = xyzdata[index, :, 1]

        xydata_current = copy.deepcopy(xyzdata[index])

        l, = plt.plot(xdata, ydata, lw=2, color='red')
        self.arrange.ax_scale(xdata, ydata, ax)
        plt.xlabel("Wavenumber (cm-1)")
        plt.ylabel("Intensity")
        title = plt.title("Spectrum")

        axcolor = 'lightgoldenrodyellow'

        # Plot index scale
        axindex = plt.axes(self.arrange.slinder1_shape, facecolor=axcolor)
        sindex_max = lens + 0.8 if lens != 1 else 1
        sindex = Slider(axindex, 'Index', 1, sindex_max,
                        valinit=index + 1, valfmt="%i")

        # Cut scale
        x_left_edge = xyzdata[0, 0, 0] + 1
        x_right_edge = xyzdata[0, -1, 0] - 1

        axindex2 = plt.axes(self.arrange.slinder2_shape, facecolor=axcolor)
        sindex2 = Slider(axindex2, 'Left edge', x_left_edge,
                         x_right_edge, valinit=x_left_edge)

        axindex3 = plt.axes(self.arrange.slinder3_shape, facecolor=axcolor)
        sindex3 = Slider(axindex3, 'Right edge', x_left_edge,
                         x_right_edge, valinit=x_right_edge)

        axindex4 = plt.axes(self.arrange.slinder4_shape, facecolor=axcolor)
        sindex4 = Slider(axindex4, 'Smooth', 0, 50, valinit=0, valfmt="%i")

        def update(val):
            nonlocal xydata_current
            index_spec = int(sindex.val) - 1

            cut_center = (sindex3.val + sindex2.val) / 2
            cut_halfwidth = (sindex3.val - sindex2.val) / 2

            if cut_halfwidth <= 0:
                l.set_xdata(0)
                l.set_ydata(0)
            else:
                data_cut = Math.cut_data(
                    xyzdata[index_spec], cut_center, cut_halfwidth)

                if sindex4.val >= 5:
                    window_size = sindex4.val
                    data_smooth = Math.smooth(
                        data_cut, window_size, polynomial_order=3)
                else:
                    data_smooth = data_cut

                xydata_current = data_smooth
                xdata = data_smooth[:, 0]
                l.set_xdata(xdata)
                ydata = data_smooth[:, 1]
                l.set_ydata(ydata)

                if radio.value_selected == 'Rescale on':
                    self.arrange.ax_scale(xdata, ydata, ax)

            fig.canvas.draw_idle()

        sindex.on_changed(update)
        sindex2.on_changed(update)
        sindex3.on_changed(update)
        sindex4.on_changed(update)

        # Reset button
        resetax = plt.axes(self.arrange.button65_shape)
        button = pltButton(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        def reset(event):
            sindex.reset()
            sindex2.reset()
            sindex3.reset()
            sindex4.reset()

        button.on_clicked(reset)

        # Export current button

        export_current_ax = plt.axes(self.arrange.button2_shape)
        button2 = pltButton(export_current_ax, 'Export current',
                            color=axcolor, hovercolor='0.975')

        def export_current(event):
            index_spec = int(sindex.val) - 1
            cut_center = (sindex3.val + sindex2.val) / 2
            cut_halfwidth = (sindex3.val - sindex2.val) / 2

            if cut_halfwidth <= 0:
                return None

            xydata = xyzdata[index_spec]
            data_cut = Math.cut_data(xydata, cut_center, cut_halfwidth)

            if sindex4.val >= 5:
                window_size = sindex4.val
                data_smooth = Math.smooth(
                    data_cut, window_size, polynomial_order=3)
            else:
                data_smooth = data_cut

            prefix = "{}_spectrum {:.0f}".format(
                self._get_prefix(**kargs), index_spec+1)

            self.io.export_file(data_smooth, default_prefix=prefix)

        button2.on_clicked(export_current)

        # Export all button
        export_all_ax = plt.axes(self.arrange.button3_shape)
        button3 = pltButton(export_all_ax, 'Export all',
                            color=axcolor, hovercolor='0.975')

        def export_all(event):
            index_spec = int(sindex.val) - 1
            cut_center = (sindex3.val + sindex2.val) / 2
            cut_halfwidth = (sindex3.val - sindex2.val) / 2

            if cut_halfwidth <= 0:
                return None

            data_cut = Math.cut_data(xyzdata, cut_center, cut_halfwidth)

            if sindex4.val >= 5:
                window_size = sindex4.val
                data_smooth = Math.smooth(
                    data_cut, window_size, polynomial_order=3)
            else:
                data_smooth = data_cut

            prefix = "{}_spectra".format(self._get_prefix(**kargs))

            self.io.export_files(data=data_smooth, prefix=prefix)

        button3.on_clicked(export_all)

        ax4 = plt.axes(self.arrange.button5_shape)
        button4 = pltButton(ax4, 'Change title',
                            color=axcolor, hovercolor='0.975')

        def change_title(event):
            try:
                plt.setp(title, text=Ask().string("Title ?"))
            except CancelInterrupt:
                return None
            fig.canvas.draw_idle()

        button4.on_clicked(change_title)

        ax6 = plt.axes(self.arrange.button6_shape)
        button6 = pltButton(ax6, 'Find peaks',
                            color=axcolor, hovercolor='0.975')

        ax7 = plt.axes(self.arrange.button7L_shape)
        button7 = pltButton(ax7, '-',
                            color=axcolor, hovercolor='0.975')

        def sub_left(event):
            self._sub_slider(sindex2)
        button7.on_clicked(sub_left)

        ax8 = plt.axes(self.arrange.button7R_shape)
        button8 = pltButton(ax8, '+',
                            color=axcolor, hovercolor='0.975')

        def add_left(event):
            self._add_slider(sindex2)
            fig.canvas.draw_idle()

        button8.on_clicked(add_left)

        ax9 = plt.axes(self.arrange.button8L_shape)
        button9 = pltButton(ax9, '-',
                            color=axcolor, hovercolor='0.975')

        def sub_right(event):
            self._sub_slider(sindex3)

        button9.on_clicked(sub_right)

        ax10 = plt.axes(self.arrange.button8R_shape)
        button10 = pltButton(ax10, '+',
                             color=axcolor, hovercolor='0.975')

        def add_right(event):
            self._add_slider(sindex3)
            fig.canvas.draw_idle()

        button10.on_clicked(add_right)

        def report_peaks(event):
            peak_report = Math.report_highest_peaks(xydata_current)
            if not peak_report:
                peak_report = "No peaks found."

            Ask.show_message_box("Message Box", peak_report)

        button6.on_clicked(report_peaks)

        # Rescale radiobuttons
        rax = plt.axes(self.arrange.radiobutton1_shape, facecolor=axcolor)
        radio = RadioButtons(rax, ('Rescale on', 'Rescale off'), active=0)

        plt.show()

    def waterfall(self, **kargs):
        # known bug: plot not working when x pixel or y pixel is 1

        pixel_x = self._get_pixel_x(**kargs)
        color = self._get_color(**kargs)
        distance = self._get_distance(**kargs)
        windows_title = self._get_windows_title()

        xyzdata = self._get_data()
        # data_lineprofile = Math.get_waterfall_data(xyzdata)

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(windows_title)
        self.arrange.adjust_figure_position(fig)

        x, y, data_lineprofile = Math.get_waterfall_xyz(xyzdata, distance)
        z = data_lineprofile

        h = ax.pcolormesh(x, y, z, cmap=color, shading='nearest')

        cb = plt.colorbar(h, label="Intensity")
        cmin = np.amin(data_lineprofile)
        cmax = np.amax(data_lineprofile)
        h.set_clim(vmin=cmin, vmax=cmax)

        plt.title("Waterfall")
        plt.xlabel("Wavenumber (cm-1)")
        if distance:
            plt.ylabel("Y (nm)")
        else:
            self._set_yaxis_int(ax)
            plt.ylabel("Y")

        axcolor = 'lightgoldenrodyellow'

        x_left_edge = xyzdata[0, 0, 0] + 1
        x_right_edge = xyzdata[0, -1, 0] - 1

        axindex2 = plt.axes(self.arrange.slinder3_shape, facecolor=axcolor)
        sindex2 = Slider(axindex2, 'Cut left', x_left_edge,
                         x_right_edge, valinit=x_left_edge)

        axindex3 = plt.axes(self.arrange.slinder4_shape, facecolor=axcolor)
        sindex3 = Slider(axindex3, 'Cut right', x_left_edge,
                         x_right_edge, valinit=x_right_edge)

        def update(val):
            nonlocal h, cmin, cmax, cb
            cut_left = sindex2.val
            cut_right = sindex3.val
            cut_center = (cut_right + cut_left) / 2
            cut_halfwidth = (cut_right - cut_left) / 2
            if cut_halfwidth <= 0:
                return None

            data = Math.cut_data(xyzdata, cut_center, cut_halfwidth)

            x, y, z = Math.get_waterfall_xyz(data, distance)
            data_lineprofile = z

            ax.clear()
            h = ax.pcolormesh(x, y, z, cmap=color, shading='nearest')
            ax.set_title("Waterfall")
            ax.set_xlabel("Wavenumber (cm-1)")

            if distance:
                ax.set_ylabel("Y (nm)")
            else:
                self._set_yaxis_int(ax)
                ax.set_ylabel("Y")

            if radio.value_selected == 'Rescale on':
                cmin = np.amin(data_lineprofile)
                cmax = np.amax(data_lineprofile)

            h.set_clim(vmin=cmin, vmax=cmax)
            cb.remove()
            cb = plt.colorbar(h, ax=ax, label="Intensity")

            fig.canvas.draw_idle()

        sindex2.on_changed(update)
        sindex3.on_changed(update)

        resetax = plt.axes(self.arrange.button1_shape)
        button = pltButton(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        def reset(event):
            sindex2.reset()
            sindex3.reset()

        button.on_clicked(reset)

        # Export current button
        export_current_ax = plt.axes(self.arrange.button3_shape)
        button2 = pltButton(export_current_ax, 'Export',
                            color=axcolor, hovercolor='0.975')

        def export(event):
            cut_left = sindex2.val
            cut_right = sindex3.val
            cut_center = (cut_right + cut_left) / 2
            cut_halfwidth = (cut_right - cut_left) / 2
            if cut_halfwidth <= 0:
                return None

            data = Math.cut_data(xyzdata, cut_center, cut_halfwidth)
            output = Math.get_waterfall_export(data, distance)

            prefix = "{}_waterfall from {:.1f} cm-1 to {:.1f} cm-1".format(
                self._get_prefix(**kargs), cut_left, cut_right)

            self.io.export_file(data=output, default_prefix=prefix)

        button2.on_clicked(export)

        rax = plt.axes(self.arrange.radiobutton1_shape, facecolor=axcolor)
        radio = RadioButtons(rax, ('Rescale on', 'Rescale off'), active=0)

        ax7 = plt.axes(self.arrange.button7L_shape)
        button7 = pltButton(ax7, '-',
                            color=axcolor, hovercolor='0.975')

        def substract(event):
            self._sub_slider(sindex2)
        button7.on_clicked(substract)

        ax8 = plt.axes(self.arrange.button7R_shape)
        button8 = pltButton(ax8, '+',
                            color=axcolor, hovercolor='0.975')

        def add(event):
            self._add_slider(sindex2)
            fig.canvas.draw_idle()

        button8.on_clicked(add)

        ax9 = plt.axes(self.arrange.button8L_shape)
        button9 = pltButton(ax9, '-',
                            color=axcolor, hovercolor='0.975')

        def substract(event):
            self._sub_slider(sindex3)
        button9.on_clicked(substract)

        ax10 = plt.axes(self.arrange.button8R_shape)
        button10 = pltButton(ax10, '+',
                             color=axcolor, hovercolor='0.975')

        def add(event):
            self._add_slider(sindex3)
            fig.canvas.draw_idle()

        button10.on_clicked(add)

        plt.show()
        return data_lineprofile

    def point_mapping(self, center=None, **kargs):
        pixel_x = self._get_pixel_x(**kargs)
        color = self._get_color(**kargs)
        distance = self._get_distance(**kargs)
        windows_title = self._get_windows_title()
        xyzdata = self._get_data()
        center = self._check_center(center)

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(windows_title)
        self.arrange.adjust_figure_position()
        title = plt.title('Point Mapping based on {:.1f} cm-1'.format(center))

        # Plot colormap
        x, y, matrix_mapping, shading = Math.get_point_mapping_xyz(
            xyzdata, center, pixel_x, distance)
        h = plt.pcolormesh(x, y, matrix_mapping, cmap=color, shading=shading)
        self.arrange.map_scale(matrix_mapping, h)

        # Add colorbar and labels
        plt.colorbar(h, label="Intensity")
        if distance:
            plt.xlabel("X (nm)")
            plt.ylabel("Y (nm)")
        else:
            self._set_xaxis_int(ax)
            self._set_yaxis_int(ax)
            plt.xlabel("X")
            plt.ylabel("Y")

        axcolor = 'lightgoldenrodyellow'

        xmin = xyzdata[0, 0, 0] + 1
        xmax = xyzdata[0, -1, 0] - 1

        # Plot index scale
        axindex1 = plt.axes(self.arrange.slinder2_shape, facecolor=axcolor)
        sindex1 = Slider(axindex1, 'Center (cm-1)', xmin, xmax, valinit=center)

        # Cut scale
        x_left_edge = xyzdata[0, 0, 0] + 1
        x_right_edge = xyzdata[0, -1, 0] - 1

        axindex2 = plt.axes(self.arrange.slinder3_shape, facecolor=axcolor)
        sindex2 = Slider(axindex2, 'Cut left', x_left_edge,
                         x_right_edge, valinit=x_left_edge)
        axindex3 = plt.axes(self.arrange.slinder4_shape, facecolor=axcolor)
        sindex3 = Slider(axindex3, 'Cut right', x_left_edge,
                         x_right_edge, valinit=x_right_edge)

        def update(val):
            center = sindex1.val
            matrix_mapping = Math.get_point_mapping_data(
                xyzdata, center, pixel_x)
            h.set_array(matrix_mapping)
            if radio.value_selected == 'Rescale on':
                self.arrange.map_scale(matrix_mapping, h)

            plt.setp(
                title, text='Point Mapping based on {:.1f} cm-1'.format(center))
            sindex1.ax.set_xlim(sindex2.val, sindex3.val)
            fig.canvas.draw_idle()

        sindex1.on_changed(update)
        sindex2.on_changed(update)
        sindex3.on_changed(update)

        resetax = plt.axes(self.arrange.button1_shape)
        button = pltButton(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        def reset(event):
            sindex1.reset()
            sindex2.reset()
            sindex3.reset()

        button.on_clicked(reset)

        # Export current button
        export_current_ax = plt.axes(self.arrange.button3_shape)
        button2 = pltButton(export_current_ax, 'Export',
                            color=axcolor, hovercolor='0.975')

        def export(event):
            center = sindex1.val
            output = Math.get_point_mapping_export(
                xyzdata, center, pixel_x, distance)
            if output is None:
                return None

            prefix = "{}_Point Mapping {:.1f} cm-1".format(
                self._get_prefix(**kargs), center)

            self.io.export_file(data=output,
                                default_prefix=prefix)

        button2.on_clicked(export)

        rax = plt.axes(self.arrange.radiobutton1_shape, facecolor=axcolor)
        radio = RadioButtons(rax, ('Rescale on', 'Rescale off'), active=0)

        ax7 = plt.axes(self.arrange.button7L_shape)
        button7 = pltButton(ax7, '-',
                            color=axcolor, hovercolor='0.975')

        def sub(event):
            self._sub_slider(sindex1)
        button7.on_clicked(sub)

        ax8 = plt.axes(self.arrange.button7R_shape)
        button8 = pltButton(ax8, '+',
                            color=axcolor, hovercolor='0.975')

        def add(event):
            self._add_slider(sindex1)
            fig.canvas.draw_idle()

        button8.on_clicked(add)

        plt.show()
        return Math.get_point_mapping_data(xyzdata, sindex1.val, pixel_x)

    def point_lineprofile(self, center=None, **kargs):
        pixel_x = 1
        color = self._get_color(**kargs)
        distance = self._get_distance(**kargs)
        windows_title = self._get_windows_title()

        xyzdata = self._get_data()
        center = self._check_center(center)

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(windows_title)
        self.arrange.adjust_figure_position()
        title = plt.title('Point Mapping based on {:.1f} cm-1'.format(center))

        xdata, ydata = Math.get_point_lineprofile_xy(xyzdata, center, distance)
        l, = plt.plot(xdata, ydata, lw=2, color='red')

        self.arrange.ax_scale(xdata, ydata, ax)
        if distance:
            plt.xlabel("Distance (nm)")
        else:
            self._set_xaxis_int(ax)
            plt.xlabel("X")

        plt.ylabel("Intensity")

        axcolor = 'lightgoldenrodyellow'

        # Cut scale
        x_left_edge = xyzdata[0, 0, 0] + 1
        x_right_edge = xyzdata[0, -1, 0] - 1
        x_middle = (x_left_edge + x_right_edge) / 2

        # Plot index scale
        axindex = plt.axes(self.arrange.slinder1_shape, facecolor=axcolor)
        sindex = Slider(axindex, 'Center wavenumber',
                        x_left_edge, x_right_edge, valinit=center)

        axindex2 = plt.axes(self.arrange.slinder2_shape, facecolor=axcolor)
        sindex2 = Slider(axindex2, 'Left edge', x_left_edge,
                         x_right_edge, valinit=x_left_edge)
        axindex3 = plt.axes(self.arrange.slinder3_shape, facecolor=axcolor)
        sindex3 = Slider(axindex3, 'Right edge', x_left_edge,
                         x_right_edge, valinit=x_right_edge)

        def update(val):
            center_wavenumber = sindex.val
            cut_halfwidth = (sindex3.val - sindex2.val) / 2

            xdata, ydata = Math.get_point_lineprofile_xy(
                xyzdata, center_wavenumber, distance)
            l.set_ydata(ydata)

            if cut_halfwidth >= 0:
                sindex.ax.set_xlim(sindex2.val, sindex3.val)

            if radio.value_selected == 'Rescale on':
                self.arrange.ax_scale(xdata, ydata, ax)

            plt.setp(
                title, text='Point Mapping based on {:.1f} cm-1'.format(center_wavenumber))
            fig.canvas.draw_idle()

        sindex.on_changed(update)
        sindex2.on_changed(update)
        sindex3.on_changed(update)

        # Reset button
        resetax = plt.axes(self.arrange.button1_shape)
        button = pltButton(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        def reset(event):
            sindex.reset()
            sindex2.reset()
            sindex3.reset()

        button.on_clicked(reset)

        prefix = "{}_Point Lineprofile {:.1f} cm-1".format(
            self._get_prefix(**kargs), center)

        def export_current(event):
            self.io.export_file(data=Math.get_point_lineprofile_export(xyzdata, center, distance),
                                default_prefix=prefix)

        # Export button
        export_ax = plt.axes(self.arrange.button3_shape)
        button3 = pltButton(export_ax, 'Export',
                            color=axcolor, hovercolor='0.975')

        button3.on_clicked(export_current)

        # Rescale radiobuttons
        rax = plt.axes(self.arrange.radiobutton1_shape, facecolor=axcolor)
        radio = RadioButtons(rax, ('Rescale on', 'Rescale off'), active=0)

        plt.show()
        return Math.get_point_lineprofile_xy(xyzdata, center, distance)

    def more_mapping(self, center=None, halfwidth=None, **kargs):
        pixel_x = self._get_pixel_x(**kargs)
        color = self._get_color(**kargs)
        distance = self._get_distance(**kargs)
        windows_title = self._get_windows_title()

        xyzdata = self._get_data()
        center = self._check_center(center)
        halfwidth = self._check_halfwidth(halfwidth)

        method = 'smart'
        matrix_mapping = Math.get_integration_mapping_data(
            xyzdata, center, halfwidth, pixel_x, method='Peak finding & Gaussian Fit')

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(windows_title)
        self.arrange.adjust_figure_position()
        title = plt.title(
            'Integration Mapping from {:.1f} to {:.1f} cm-1'.format(center-halfwidth, center+halfwidth))

        # Get integration mapping data
        lens = len(xyzdata)
        pixel_y = int(lens / pixel_x)
        x, y, shading = Math.get_mapping_axes(distance, pixel_x, pixel_y)

        matrix_mapping = Math.get_integration_mapping_data(
            xyzdata, center, halfwidth, pixel_x, method=method)

        h = plt.pcolormesh(x, y, matrix_mapping, cmap=color, shading=shading)
        self._set_yaxis_int(ax)

        self.arrange.map_scale(matrix_mapping, h)
        colorbar1 = plt.colorbar(h, label="Intensity")
        if distance:
            plt.xlabel("X (nm)")
            plt.ylabel("Y (nm)")
        else:
            self._set_xaxis_int(ax)
            self._set_yaxis_int(ax)
            plt.xlabel("X")
            plt.ylabel("Y")

        x_left_edge = xyzdata[0, 0, 0] + 1
        x_right_edge = xyzdata[0, -1, 0] - 1
        axcolor = 'lightgoldenrodyellow'

        axindex = plt.axes(self.arrange.slinder1_shape, facecolor=axcolor)
        sindex1 = Slider(axindex, 'Left (cm-1)', x_left_edge,
                         x_right_edge, valinit=center - halfwidth)

        axindex2 = plt.axes(self.arrange.slinder2_shape, facecolor=axcolor)
        sindex2 = Slider(axindex2, 'Right (cm-1)', x_left_edge,
                         x_right_edge, valinit=center + halfwidth)

        axindex3 = plt.axes(self.arrange.slinder3_shape, facecolor=axcolor)
        sindex3 = Slider(axindex3, 'Cut Left', x_left_edge,
                         x_right_edge, valinit=x_left_edge)

        axindex4 = plt.axes(self.arrange.slinder4_shape, facecolor=axcolor)
        sindex4 = Slider(axindex4, 'Cut right', x_left_edge,
                         x_right_edge, valinit=x_right_edge)

        error_exception = None

        def update(val):
            nonlocal matrix_mapping
            nonlocal error_exception
            new_center = (sindex2.val + sindex1.val) / 2
            new_halfwidth = (sindex2.val - sindex1.val) / 2
            new_method = radio3.value_selected
            if new_halfwidth > 0:
                sindex1.ax.set_xlim(sindex3.val, sindex4.val)
                sindex2.ax.set_xlim(sindex3.val, sindex4.val)
            else:
                return None

            try:
                typ = radio2.value_selected
                if typ == 'Integrate':
                    plt.setp(title, text='Integration Mapping from {:.1f} to {:.1f} cm-1'.format(
                        new_center-new_halfwidth, new_center+new_halfwidth))
                    matrix_mapping = Math.get_integration_mapping_data(
                        xyzdata, new_center, new_halfwidth, pixel_x, new_method)
                elif typ == 'Shift':
                    plt.setp(title, text='Shift Mapping')
                    if new_method == 'normal':
                        matrix_mapping, median_wavenumber = Math.get_shift_mapping_data(
                            xyzdata, new_center, new_halfwidth, pixel_x, 'normal')
                    else:
                        matrix_mapping, median_wavenumber = Math.get_shift_mapping_data(
                            xyzdata, new_center, new_halfwidth, pixel_x, 'super smart')
                    plt.setp(
                        title, text='Shift Mapping based on {:.1f} cm-1'.format(median_wavenumber))
                elif typ == 'FWHM':
                    plt.setp(title, text='Full Wave at Half Maximum Mapping')
                    matrix_mapping = Math.get_sigma_mapping_data(
                        xyzdata, new_center, new_halfwidth, pixel_x, new_method)

            except Exception as ex:
                button3.color = 'red'
                button3.hovercolor = 'red'
                error_exception = ex
            else:
                button3.color = 'white'
                button3.hovercolor = 'white'
                error_exception = None

            h.set_array(matrix_mapping)
            if radio.value_selected == 'Rescale on':
                self.arrange.map_scale(matrix_mapping, h)

            if radio2.value_selected == 'Integrate':
                colorbar1.ax.set_ylabel('Intensity')
            else:
                colorbar1.ax.set_ylabel('Wavenumber (cm-1)')

            fig.canvas.draw_idle()

        sindex1.on_changed(update)
        sindex2.on_changed(update)
        sindex3.on_changed(update)
        sindex4.on_changed(update)

        resetax = plt.axes(self.arrange.button6_shape)
        button = pltButton(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        def reset(event):
            sindex1.reset()
            sindex2.reset()
            sindex3.reset()
            sindex4.reset()

        button.on_clicked(reset)

        resetax2 = plt.axes(self.arrange.button7_shape)
        button2 = pltButton(resetax2, 'Export',
                            color=axcolor, hovercolor='0.975')

        def export(event):
            if error_exception:
                return None

            x, y, shading = Math.get_mapping_axes(
                distance, pixel_x, pixel_y, export=True)
            typ = radio2.value_selected
            left = sindex1.val
            right = sindex2.val
            prefix = "{}_{} Mapping from {:.1f} cm-1 to {:.1f} cm-1".format(
                self._get_prefix(**kargs), typ, left, right)

            self.io.export_file(data=Math.sheet_export(x, y, matrix_mapping),
                                default_prefix=prefix)

        button2.on_clicked(export)

        resetax3 = plt.axes(self.arrange.button0_shape)
        button3 = pltButton(resetax3, 'State',
                            color='white', hovercolor='white')

        def show_error(event):
            if error_exception is None:
                return None
            else:
                error_message_box(error_exception, 'Mapping')

        button3.on_clicked(show_error)

        rax = plt.axes(self.arrange.radiobutton1_shape, facecolor=axcolor)
        radio = RadioButtons(rax, ('Rescale on', 'Rescale off'), active=0)

        rax2 = plt.axes(self.arrange.radiobutton2_shape, facecolor=axcolor)
        radio2 = RadioButtons(rax2, ('Integrate', 'Shift', 'FWHM'), active=0)

        radio2.on_clicked(update)

        radio3_label1 = tkinter.StringVar(value='normal')
        radio3_label2 = tkinter.StringVar(value='smart')

        rax3 = plt.axes(self.arrange.radiobutton3_shape, facecolor=axcolor)
        radio3 = RadioButtons(
            rax3, (radio3_label1.get(), radio3_label2.get()), active=0)

        radio3.on_clicked(update)

        plt.show()

        return Math.get_integration_mapping_data(xyzdata, center, halfwidth, pixel_x, method='Peak finding & Gaussian Fit')

    def more_lineprofile(self, center=None, halfwidth=10, **kargs):
        color = self._get_color(**kargs)
        distance = self._get_distance(**kargs)
        windows_title = self._get_windows_title()
        pixel_x = 1
        xyzdata = self._get_data()
        center = self._check_center(center)
        halfwidth = self._check_halfwidth(halfwidth)

        matrix_mapping = Math.get_integration_mapping_data(
            xyzdata, center, halfwidth, pixel_x, method='Peak finding & Gaussian Fit')

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(windows_title)
        self.arrange.adjust_figure_position()
        title = plt.title(
            'Integration Mapping from {:.1f} to {:.1f} cm-1'.format(center-halfwidth, center+halfwidth))

        x_left_edge = xyzdata[0, 0, 0] + 1
        x_right_edge = xyzdata[0, -1, 0] - 1
        axcolor = 'lightgoldenrodyellow'

        error_exception = None
        lens = len(xyzdata)

        if distance:
            xdata = np.flipud(np.arange(0, lens*distance, distance))
        else:
            xdata = np.flipud(np.arange(0, lens, 1))

        ydata = matrix_mapping[:, 0]
        l, = plt.plot(xdata, ydata, lw=2, color='red')
        self.arrange.ax_scale(xdata, ydata, ax)

        if distance:
            plt.xlabel("Distance (nm)")
        else:
            self._set_xaxis_int(ax)
            self._set_yaxis_int(ax)
            plt.xlabel("X")

        plt.ylabel("Intensity")

        axcolor = 'lightgoldenrodyellow'

        axindex = plt.axes(self.arrange.slinder1_shape, facecolor=axcolor)
        sindex1 = Slider(axindex, 'Left (cm-1)', x_left_edge,
                         x_right_edge, valinit=center - halfwidth)

        axindex2 = plt.axes(self.arrange.slinder2_shape, facecolor=axcolor)
        sindex2 = Slider(axindex2, 'Right (cm-1)', x_left_edge,
                         x_right_edge, valinit=center + halfwidth)

        axindex3 = plt.axes(self.arrange.slinder3_shape, facecolor=axcolor)
        sindex3 = Slider(axindex3, 'Cut Left', x_left_edge,
                         x_right_edge, valinit=x_left_edge)

        axindex4 = plt.axes(self.arrange.slinder4_shape, facecolor=axcolor)
        sindex4 = Slider(axindex4, 'Cut right', x_left_edge,
                         x_right_edge, valinit=x_right_edge)

        def update(val):
            nonlocal matrix_mapping
            nonlocal error_exception
            new_center = (sindex2.val + sindex1.val) / 2
            new_halfwidth = (sindex2.val - sindex1.val) / 2
            new_method = radio3.value_selected
            if new_halfwidth > 0:
                sindex1.ax.set_xlim(sindex3.val, sindex4.val)
                sindex2.ax.set_xlim(sindex3.val, sindex4.val)
            else:
                return None

            try:
                typ = radio2.value_selected
                if typ == 'Integrate':
                    plt.setp(title, text='Integration Mapping from {:.1f} to {:.1f} cm-1'.format(
                        new_center-new_halfwidth, new_center+new_halfwidth))
                    matrix_mapping = Math.get_integration_mapping_data(
                        xyzdata, new_center, new_halfwidth, pixel_x, new_method)
                    ax.set_ylabel("Intensity")

                elif typ == 'Shift':
                    plt.setp(title, text='Shift Mapping')
                    matrix_mapping, median_wavenumber = Math.get_shift_mapping_data(
                        xyzdata, new_center, new_halfwidth, pixel_x, new_method)

                    plt.setp(
                        title, text='Shift Mapping based on {:.1f} cm-1'.format(median_wavenumber))
                    ax.set_ylabel("Wavenumber Shift (cm-1)")

                elif typ == 'FWHM':
                    plt.setp(title, text='Full Wave at Half Maximum Mapping')
                    matrix_mapping = Math.get_sigma_mapping_data(
                        xyzdata, new_center, new_halfwidth, pixel_x, new_method)
                    ax.set_ylabel("FWHM (cm-1)")

            except Exception as ex:
                button3.color = 'red'
                button3.hovercolor = 'red'
                error_exception = ex
            else:
                button3.color = 'white'
                button3.hovercolor = 'white'
                error_exception = None

            ydata = matrix_mapping[:, 0]
            l.set_ydata(ydata)
            if radio.value_selected == 'Rescale on':
                self.arrange.ax_scale(xdata, ydata, ax)

            fig.canvas.draw_idle()

        sindex1.on_changed(update)
        sindex2.on_changed(update)
        sindex3.on_changed(update)
        sindex4.on_changed(update)

        resetax = plt.axes(self.arrange.button6_shape)
        button = pltButton(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        def reset(event):
            sindex1.reset()
            sindex2.reset()
            sindex3.reset()
            sindex4.reset()

        button.on_clicked(reset)

        resetax2 = plt.axes(self.arrange.button7_shape)
        button2 = pltButton(resetax2, 'Export',
                            color=axcolor, hovercolor='0.975')

        def export(event):
            if matrix_mapping is None:
                return None

            if distance:
                xdata = np.flipud(np.arange(0, lens*distance, distance))
            else:
                xdata = np.flipud(np.arange(0, lens, 1))
            output = np.c_[xdata, matrix_mapping]
            typ = radio2.value_selected
            left = sindex1.val
            right = sindex2.val
            prefix = "{}_{} Lineprofile from {:.1f} cm-1 to {:.1f} cm-1".format(
                self._get_prefix(**kargs), typ, left, right)

            self.io.export_file(data=output,
                                default_prefix=prefix)

        button2.on_clicked(export)

        resetax3 = plt.axes(self.arrange.button0_shape)
        button3 = pltButton(resetax3, 'State',
                            color='white', hovercolor='white')

        def show_error(event):
            if error_exception is None:
                return None
            else:
                error_message_box(error_exception, 'Mapping')

        button3.on_clicked(show_error)

        rax = plt.axes(self.arrange.radiobutton1_shape, facecolor=axcolor)
        radio = RadioButtons(rax, ('Rescale on', 'Rescale off'), active=0)

        rax2 = plt.axes(self.arrange.radiobutton2_shape, facecolor=axcolor)
        radio2 = RadioButtons(rax2, ('Integrate', 'Shift', 'FWHM'), active=0)

        radio2.on_clicked(update)

        radio3_label1 = tkinter.StringVar(value='normal')
        radio3_label2 = tkinter.StringVar(value='smart')

        rax3 = plt.axes(self.arrange.radiobutton3_shape, facecolor=axcolor)
        radio3 = RadioButtons(
            rax3, (radio3_label1.get(), radio3_label2.get()), active=0)

        radio3.on_clicked(update)

        plt.show()

        return Math.get_integration_mapping_data(xyzdata, center, halfwidth, pixel_x, method='Peak finding & Gaussian Fit')

