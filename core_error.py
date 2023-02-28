from inspect import Traceback
import tkinter.messagebox as messagebox
from core_logger import logger, silent_logger


def error_message_box(error=None, method=''):
    if method == '':
        message_output = "Something went wrong."
    else:
        message_output = "{} went wrong.".format(method)

    if error is not None:
        if error.args == '':
            template = "\nAn exception of type {0} occurred."
            message_output += template.format(type(error).__name__)
        elif len(error.args) == 1:
            template = "\nAn exception of type {0} occurred.\n{1}"
            message_output += template.format(type(error).__name__, error.args)
        elif len(error.args) >= 2:
            template = "\nAn exception of type {0} occurred.\n{1}"
            message_output += template.format(
                type(error).__name__, error.args[1])

    messagebox.showinfo("Error", message_output)


class CancelInterrupt(Exception):
    pass


class CustomError():
    def __init__(self, error: Exception, method: str = None) -> None:
        self.method = method
        self.error = error
        self.message_output = self._get_message_output(method)
        self.error_output = self._get_error_output(error)

    def _get_message_output(self, method=None):
        if method is not None and isinstance(method, str):
            if method == '':
                message_output = "Something went wrong."
            elif isinstance(method, str):
                message_output = "{} went wrong.".format(method)
            else:
                message_output = "Error tracing went wrong."
        else:
            try:
                st = Traceback.extract_stack()
                a = st[-3]
                message_output = "{}, line {} in {} went wrong.".format(
                    a[0], a[1], a[2])
            except:
                message_output = "Something went wrong."

        if not message_output:
            message_output = ''
        return message_output

    def _get_error_output(self, error):
        if error is not None:
            if not error.args or error.args == '':
                template = "An exception of type {0} occurred."
                error_output = template.format(type(error).__name__)
            elif len(error.args) == 1:
                template = "An exception of type {0} occurred.\n{1}"
                error_output = template.format(
                    type(error).__name__, error.args)
            else:
                template = "An exception of type {0} occurred.\n{1}"
                error_output = template.format(
                    type(error).__name__, error.args[1])
        else:
            error_output = ''

        if not error_output:
            error_output = ''
        return error_output

    def message_box(self):
        message_output, error_output = self.message_output, self.error_output
        messagebox.showinfo("Error", "{}\n{}".format(
            message_output, error_output))

    def debug(self):
        if isinstance(self.error, CancelInterrupt):
            return None

        message_output, error_output = self.message_output, self.error_output
        if error_output is not None and error_output != '':
            logger.debug(message_output.replace('\n', ' '))
            logger.debug(error_output.replace('\n', ' '))
        else:
            logger.error("Error")
        return -1

    def show_error(self):
        if isinstance(self.error, CancelInterrupt):
            return None

        message_output, error_output = self.message_output, self.error_output
        if error_output is not None and error_output != '':
            messagebox.showerror(message=message_output+error_output)
        else:
            messagebox.showerror(message="Error")
        return -1
