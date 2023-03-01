from src.utils.core_repository import Repository
from src.ters.core_TERS import TERS
from src.utils.core_setting import Setting


class MultiTERS():
    def __init__(self, file_cal: str = None, file_ori: list = [], settings: Setting = None, **kargs) -> None:
        self.settings = settings
        self.repo = Repository()
        self.properties = kargs
        self.properties['filenames'] = file_ori
        self.default_properties = {'pixel_x': 1,
                                   'distance': 0}
        self.repo.set_data([])

        for item in file_ori:
            self.repo.data.append(TERS(file_cal=file_cal,
                                       file_ori=item,
                                       settings=settings,
                                       **kargs))

    def export_all_no_options(self):
        for item in self.repo.data:
            item: TERS
            item.export_all(silence=True)

    def merge_to_one_and_export(self):
        for item in self.repo.data:
            item: TERS
            item.math.merge_to_one()
        return self.export_all_no_options()
