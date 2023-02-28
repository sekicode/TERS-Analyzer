import json
import pathlib


class Setting():
    '''Store settings between application and files'''

    def __init__(self, file, default_settings={}) -> None:
        self.settings = {}
        self.file = file
        self.default_file = 'Default Settings.json'
        self.indent = 4

        self.default_settings = default_settings if default_settings else {}

        self._load()

    def _write(self):
        '''Update `self.setting` to files'''
        with open(self.file, 'w') as load_f:
            json.dump(self.settings, load_f, indent=self.indent)

    def _load(self):
        '''Load file to `self.setting`'''
        path = pathlib.Path(self.file)
        if path.exists():
            try:
                with open(self.file, 'r') as load_f:
                    self.settings = json.load(load_f)
                    self.set_default(force=False)
            except json.JSONDecodeError:
                self.set_default()
        else:
            open(self.file, 'a').close()
            self.set_default()

    def set(self, **kargs):
        '''Set properties'''
        for key, value in kargs.items():
            self.settings[key] = value
        # self._renew()
        self._write()

    def get(self, *args):
        '''Get properties'''
        ret = []
        for item in args:
            ret.append(self.settings.get(item))
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def clear(self):
        self.settings = {}
        self._write()

    def set_default(self, *args, force=True):
        # Try to load 'Default_Settings.json' if existed
        try:
            with open(self.default_file, 'r') as load_f:
                self.default_settings = json.load(load_f)
        except:
            with open(self.default_file, 'a') as load_f:
                json.dump({}, load_f, indent=self.indent)

        # if force==True: set all to default
        # if force==False: add default
        if not args:
            if force:
                self.clear()
                if hasattr(self, 'default_settings'):
                    self.set(**self.default_settings)
            else:
                for key, value in self.default_settings.items():
                    if self.settings.get(key) is None:
                        self.settings[key] = value
        else:
            for item in args:
                try:
                    if force:
                        self.settings[item] = self.default_settings[item]
                    else:
                        for key, value in self.default_settings[item].items():
                            if self.settings[item].get(key) is None:
                                self.settings[item][key] = value
                except:
                    pass

        self._write()
