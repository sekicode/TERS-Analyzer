import matplotlib.pyplot as plt
import numpy as np


class PlotArrange():
    def __init__(self) -> None:
        self.slinder1_shape = [0.2, 0.2, 0.65, 0.03]
        self.slinder2_shape = [0.2, 0.15, 0.65, 0.03]
        self.slinder3_shape = [0.2, 0.1, 0.65, 0.03]
        self.slinder4_shape = [0.2, 0.05, 0.65, 0.03]

        self.button0_shape = [0.025, 0.9, 0.2, 0.05]
        self.button65_shape = [0.025, 0.65, 0.2, 0.05]
        self.button1_shape = [0.025, 0.6, 0.2, 0.05]
        self.button2_shape = [0.025, 0.55, 0.2, 0.05]
        self.button3_shape = [0.025, 0.5, 0.2, 0.05]
        self.button4_shape = [0.025, 0.45, 0.2, 0.05]
        self.button5_shape = [0.025, 0.4, 0.2, 0.05]
        self.button6_shape = [0.025, 0.35, 0.2, 0.05]

        self.button7_shape = [0.025, 0.3, 0.2, 0.05]
        self.button7L_shape = [0.025, 0.3, 0.1, 0.05]
        self.button7R_shape = [0.125, 0.3, 0.1, 0.05]

        self.button8_shape = [0.025, 0.25, 0.2, 0.05]
        self.button8L_shape = [0.025, 0.25, 0.1, 0.05]
        self.button8R_shape = [0.125, 0.25, 0.1, 0.05]

        self.button9_shape = [0.025, 0.2, 0.2, 0.05]
        self.button9L_shape = [0.025, 0.2, 0.1, 0.05]
        self.button9R_shape = [0.125, 0.2, 0.1, 0.05]

        self.radiobutton1_shape = [0.025, 0.75, 0.2, 0.15]
        self.radiobutton2_shape = [0.025, 0.55, 0.2, 0.2]
        self.radiobutton3_shape = [0.025, 0.40, 0.2, 0.15]

        self.textbox1_shape = [0.2, 0.2, 0.2, 0.03]
        self.textbox2_shape = [0.2, 0.15, 0.2, 0.03]

    def adjust_figure_position(self, fig=None):
        if fig:
            fig.subplots_adjust(left=0.35, bottom=0.35)
        else:
            plt.subplots_adjust(left=0.35, bottom=0.35)

    def _get_scale(self, xdata):
        max = np.amax(xdata)
        min = np.amin(xdata)
        range = max - min
        max += range * 0.1
        min -= range * 0.1
        return min, max

    def ax_scale_y(self, ydata, ax):
        ymin, ymax = self._get_scale(ydata)
        ax.set_ylim(ymin=ymin, ymax=ymax)

    def ax_scale_x(self, xdata, ax):
        xmin, xmax = self._get_scale(xdata)
        ax.set_xlim(xmin=xmin, xmax=xmax)

    def ax_scale(self, xdata, ydata, ax):
        self.ax_scale_y(ydata, ax)
        self.ax_scale_x(xdata, ax)

    def map_scale(self, matrix_mapping, quadmesh):
        cmin = np.amin(matrix_mapping)
        cmax = np.amax(matrix_mapping)
        quadmesh.set_clim(vmin=cmin, vmax=cmax)
