
from typing import Dict

import torch

from cnsproject.network.monitors import AbstractMonitor


class DummyMonitor(AbstractMonitor):
    def __init__(self, values: Dict[str, torch.Tensor]):
        self.recording = values

    def set_value(self, key: str, value: torch.Tensor):
        self.recording[key] = value

    def get(self, variable: str) -> torch.Tensor:
        return self.recording[variable]
