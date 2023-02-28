from PyQt5.QtCore import QMutex, QThread, QThreadPool
import time


class CustomThread(QThread):
    # 自定义线程类：继承 QThread
    # delay=None means no recurrences
    def __init__(self, func, delay=None, **kargs):
        QThread.__init__(self)
        # 实例化线程锁对象
        self.mutex = QMutex()
        self.func = func
        self.delay = delay
        self.result = None
        self.kargs = kargs


    def run(self):
        import debugpy
        debugpy.debug_this_thread()

        self.mutex.lock()                            # 在子线程运行时加锁
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
        """ 重写 terminate 方法 """
        self.mutex.unlock()                          # 在终止线程前先解锁
        super().terminate()   # 终止线程
        self.wait()                             # 等待线程被终止完毕

    def get_result(self):
        return self.result
