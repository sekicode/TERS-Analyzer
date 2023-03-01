from src.ters.core_my_winspec import MySpeFile
import numpy as np
import tkinter.filedialog
import tkinter.messagebox as messagebox
from io import TextIOWrapper
import os


class TERS_IO():
    def __init__(self) -> None:
        pass

    def export_file(self, data, default_prefix=None, filepath=None):
        if data is None:
            return None
        if filepath is None or filepath == '':
            if default_prefix:
                filepath = tkinter.filedialog.asksaveasfile(defaultextension=".csv",
                                                            filetypes=(
                                                                ("CSV file", "*.csv"), ("All Files", "*.*")),
                                                            initialfile=default_prefix)
            else:
                filepath = tkinter.filedialog.asksaveasfile(defaultextension=".csv",
                                                            filetypes=(("CSV file", "*.csv"), ("All Files", "*.*")))

        if isinstance(filepath, TextIOWrapper):
            filepath = filepath.name
        if filepath is None or filepath == '':
            return None
        try:
            np.savetxt(filepath, data, delimiter=',')
        except:
            messagebox.showinfo("Error", "Cannot save the file.")

    def export_files(self, data, prefix=None, folder_name=None):
        if not folder_name or data is None:
            folder_name = tkinter.filedialog.askdirectory()
        if not folder_name or data is None:
            return None
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        i = 0
        try:
            for temp in data:
                i += 1
                if prefix:
                    file_name = '{}/{}_{}.csv'.format(folder_name, prefix, i)
                else:
                    file_name = '{}/{}.csv'.format(folder_name, i)
                np.savetxt(file_name, np.asarray(temp), delimiter=',')
        except:
            messagebox.showinfo(
                "Error", "Something went wrong while exporting #{} spectrum.".format(i + 1))

    def get_winspec_file(self):
        file = tkinter.filedialog.askopenfilename()
        if file is None or file == '':
            return None
        try:
            return MySpeFile(file)
        except FileNotFoundError:
            messagebox.showinfo("Error", "Where is the file?")
        except:
            messagebox.showinfo("Error", "Files cannot be read")
