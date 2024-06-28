import os

import config
import torch
from config import CONFIG


def load_config(file):
    global CONFIG
    CONFIG = config.set_config(file)

device = torch.device(os.environ.get("DEVICE", CONFIG["device"]))
