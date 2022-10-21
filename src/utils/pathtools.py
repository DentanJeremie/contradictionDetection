from pathlib import Path

class CustomizedPath():

    def __init__(self):
        self._root = Path(__file__).parent.parent.parent

    @property
    def root(self):
        return self._root

    @property
    def data(self):
        return self.root / 'data'

    @property
    def output(self):
        return self.root / 'output'

project = CustomizedPath() 