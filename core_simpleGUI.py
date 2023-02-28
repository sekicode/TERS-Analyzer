import PySimpleGUI as sg
import tkinter.messagebox as messagebox


class MySimpleGUI:
    def __init__(self) -> None:
        pass

    def ask_spe_file(self):
        layout = [[sg.Text("Laser center wavelength (nm)?"), sg.Input()]]

        layout += [[sg.Text("Choose a fitting method")]]

        layout += [[sg.Radio("Linear", "fitting method"),
                   sg.Radio("Quadratic", "fitting method", default=True),
                   sg.Radio("Cubic", "fitting method")]]
        layout += [[sg.Button('Ok')]]

        window = sg.Window('Window Title', layout)
        success = False

        while True:
            event, values = window.read()
            laser_wavelength = values[0]
            METHOD = {1: 'linear', 2: 'quadratic', 3: 'cubic'}
            if values[1]:
                method_selected = 1
            elif values[2]:
                method_selected = 3
            elif values[3]:
                method_selected = 3

            if laser_wavelength is not None and laser_wavelength != '' and float(laser_wavelength) > 0:
                break

            messagebox.showinfo("Error", "Please enter a positive number")

        window.close()
        return laser_wavelength, method_selected


if __name__ == "__main__":
    simple_gui = MySimpleGUI()
    print(simple_gui.ask_spe_file())
