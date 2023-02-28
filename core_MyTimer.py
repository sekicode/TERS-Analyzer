# coding:utf-8
import time


class MyTimer(object):
    '''
    用上下文管理器计时
    '''

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = time.time()

        print('[finished, spent time: {time:9f} ms]'.format(
            time=(self.t1 - self.t0)*1000))
