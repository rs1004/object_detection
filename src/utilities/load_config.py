from tempfile import TemporaryDirectory, NamedTemporaryFile
from shutil import copy
from importlib import import_module
from pathlib import Path
import sys


class Config:
    def __init__(self, config_path: str):
        m = self._load(config_path)

        self.data = m.data
        self.model = m.model
        self.optimizer = m.optimizer
        self.scheduler = m.scheduler
        self.runtime = m.runtime
        self.predictor = m.predictor
        self.evaluator = m.evaluator

    def _load(self, config_path: str):
        with TemporaryDirectory() as temp_dir:
            with NamedTemporaryFile(dir=temp_dir, suffix='.py') as temp_file:
                copy(config_path, temp_file.name)
                sys.path.insert(0, temp_dir)
                module_name = Path(temp_file.name).stem
                m = import_module(module_name)
        return m
