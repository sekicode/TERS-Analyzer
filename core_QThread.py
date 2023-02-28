from PyQt5.QtCore import QMutex, QThread, QThreadPool
import time


class CustomThread(QThread):
    # delay=None means no recurrences
    def __init__(self, func, delay=None, **kargs):
        QThread.__init__(self)
        self.mutex = QMutex()
        self.func = func
        self.delay = delay
        self.result = None
        self.kargs = kargs


    def run(self):
        import debugpy
        debugpy.debug_this_thread()

        self.mutex.lock()
        if self.delay is None:
            if self.kargs:
                self.result = self.func(self.kargs)
            else:
                self.result = self.func()
        else:
            while True:
                if self.kargs:
                    self.result = self.func(self.kargs)
                else:
                    self.result = self.func()
                time.sleep(self.delay)

    def terminate(self):
        self.mutex.unlock()
        super().terminate()
        self.wait()

    def get_result(self):
        return self.result
