from core_GUI import Application, Version
from core_error import error_message_box


def main():
    try:
        app = Application()
        app.master.title('TERS Analyzer {}'.format(Version))
        app.set_window_middle(width=880, height=700)
        app.mainloop()
    except Exception as error:
        error_message_box(error)
    except:
        error_message_box()


if __name__ == "__main__":
    main()
