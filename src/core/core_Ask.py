import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from src.utils.core_error import CancelInterrupt


class Ask():
    def __init__(self) -> None:
        pass

    @classmethod
    def show_message_box(cls, titie="Message Box", message=""):
        newWin = tk.Tk()
        newWin.withdraw()
        try:
            messagebox.showinfo(titie, message, parent=newWin)
        finally:
            newWin.destroy()

    def positive_float(self, message):
        newWin = tk.Tk()
        newWin.withdraw()
        try:
            for i in range(10):
                temp = simpledialog.askfloat(
                    "Messagebox", message, parent=newWin)
                if temp is None:
                    raise CancelInterrupt
                elif temp > 0:
                    return temp
                else:
                    messagebox.showinfo(
                        "Error", "Please enter a positive number")
        finally:
            newWin.destroy()

    def float(self, message):
        newWin = tk.Tk()
        newWin.withdraw()
        try:
            for i in range(10):
                temp = simpledialog.askfloat(
                    "Messagebox", message, parent=newWin)
                if temp is None:
                    raise CancelInterrupt
                else:
                    return temp
        finally:
            newWin.destroy()

    def positive_int(self, message):
        newWin = tk.Tk()
        newWin.withdraw()
        try:
            for i in range(10):
                temp = simpledialog.askinteger(
                    "Messagebox", message, parent=newWin)
                if temp is None:
                    raise CancelInterrupt
                elif temp > 0:
                    return temp
                else:
                    messagebox.showinfo(
                        "Error", "Please enter a positive integer")
        finally:
            newWin.destroy()

    def non_negetive_int(self, message):
        newWin = tk.Tk()
        newWin.withdraw()
        try:
            for i in range(10):
                temp = simpledialog.askinteger(
                    "Messagebox", message, parent=newWin)
                if temp is None:
                    raise CancelInterrupt
                elif temp >= 0:
                    return temp
                else:
                    messagebox.showinfo(
                        "Error", "Please enter a positive integer")
        finally:
            newWin.destroy()

    def some_int(self, message, accepted_number):
        newWin = tk.Tk()
        newWin.withdraw()
        try:
            for i in range(10):
                temp = simpledialog.askinteger(
                    "Messagebox", message, parent=newWin)
                if temp is None:
                    raise CancelInterrupt
                elif temp in accepted_number:
                    return temp
                else:
                    messagebox.showinfo(
                        "Error", "Please enter a positive integer within {}".format(accepted_number))
        finally:
            newWin.destroy()

    def string(self, message):
        newWin = tk.Tk()
        newWin.withdraw()
        temp = simpledialog.askstring("Messagebox", message, parent=newWin)
        newWin.destroy()
        if temp is None or temp == '':
            raise CancelInterrupt
        return temp

    def yes_or_no(self, message):
        newWin = tk.Tk()
        newWin.withdraw()
        temp = messagebox.askyesnocancel("Messagebox", message, parent=newWin)
        newWin.destroy()
        if temp is None:
            raise CancelInterrupt
        return temp

    def folder(self):
        temp = filedialog.askdirectory()
        if temp is None or temp == '':
            raise CancelInterrupt
        else:
            return temp

    def angle(self):
        angle = self.float("Angle (deg)?") % 360
        return angle if angle < 180 else angle-360

    def center(self):
        newWin = tk.Tk()
        newWin.withdraw()
        try:
            for attempt in range(10):
                temp = simpledialog.askfloat(
                    "Messagebox", "Center wavenumber (cm-1)?")
                if temp is None:
                    return None
                elif temp > 0:
                    return temp
                else:
                    messagebox.showinfo(
                        "Error", "Please enter a positive integer")
            return None
        finally:
            newWin.destroy()
