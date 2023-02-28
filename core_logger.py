import datetime
import logging
import csv
import io


'''
Import example: from core_logger import logger
No need to import CsvFormatter
'''


class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)

    def format(self, record):
        dt = datetime.datetime.now()
        # self.writer.writerow([dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M:%S'),
        #                       record.levelname, record.name,  record.module, record.msg])
        self.writer.writerow(
            [dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M:%S'), record.module, record.msg])
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()


def get_logger():
    # Log file export
    logging.basicConfig(level=logging.DEBUG,
                        filename="{}_log.csv".format(datetime.date.today()),
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Set file export format
    logging.root.handlers[0].setFormatter(CsvFormatter())

    # Logs export to concole
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    chlr.setLevel('DEBUG')  # 也可以不设置，不设置就默认用logger的level
    logger.addHandler(chlr)

    # Set concole export format
    format_concole = logging.Formatter('[%(levelname)s] %(message)s')
    chlr.setFormatter(format_concole)

    # filename = "{}_log.csv".format(datetime.date.today())
    # fhlr = logging.FileHandler(filename)  # 输出到文件的handler
    # logger.addHandler(fhlr)
    return logger


logger = get_logger()


class SilentLogger:
    '''
    A logger ignoring only `info`
    '''
    def __init__(self, logger) -> None:
        self.logger = logger

    def info(self, *args, **kargs):
        pass

    def debug(self, *args, **kargs):
        self.logger.debug(*args, **kargs)

    def error(self, *args, **kargs):
        self.logger.error(*args, **kargs)


silent_logger = SilentLogger(logger)
