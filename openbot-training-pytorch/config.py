import json5 as json
import os


class Config(object):
    def __init__(self, config_file):
        #load config file from config.json
        self.config_file = config_file
        with open(config_file, 'r') as fp:
            data, = json.load(fp)
            self._config = data

    def get_config(self):
        return self._config

    def _load_config(self, new_config):
        with open(new_config, 'r') as fp:
            data, = json.load(fp)
            self._config = data
        return self._config

    def set_config(self, new_config):
        return self._load_config(new_config)

    def save_config(self):
        with open('config.json', 'w') as outfile:
            json.dump([self._config], outfile, indent=4, sort_keys=True)


_instance = Config(os.environ.get("CONFIG", 'config.json'))
CONFIG = _instance.get_config()
set_config = _instance.set_config
get_config = _instance.get_config

#Append this to imports
"""

import config
from config import CONFIG

def load_config(file):
    global CONFIG
    CONFIG = config.set_config(file)

"""

