from pathlib import Path
import datetime


class CustomizedPath():

    def __init__(self):
        self._root = Path(__file__).parent.parent.parent

    def get_new_bert_chepoint(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result = self.bert_checkpoints / ('checkpoint' + now)
        result.mkdir(parents=True, exist_ok = True)
        return result

    def get_newest_bert_checkpoint(self):
        directories = sorted([
            str(path)
            for path in self.bert_checkpoints.iterdir()
            if path.is_dir()
            and any(path.iterdir()) # Non-empty directory
        ])

        if len(directories) == 0:
            return None
        return Path(directories[-1]) / 'best'

    @property
    def root(self):
        return self._root

    @property
    def data(self):
        result = (self.root / 'data')
        result.mkdir(parents=True, exist_ok = True)
        return result

    @property
    def output(self):
        result = (self.root / 'output')
        result.mkdir(parents=True, exist_ok = True)
        return result

    @property
    def bert_checkpoints(self):
        result = (self.output / 'bert_checkpoints')
        result.mkdir(parents=True, exist_ok = True)
        return result

project = CustomizedPath() 