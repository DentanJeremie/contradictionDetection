import collections
import datetime
import os
from pathlib import Path
import typing as t

import pandas as pd
import requests


class CustomizedPath():

    files = {
        "train":{
            "save_as":"train.csv",
            "url":"https://drive.google.com/uc?export=download&id=1w8OU3fRFMVhbkkecJsHM4ffYRQxc20Tq",
        },
        "test":{
            "save_as":"test.csv",
            "url":"https://drive.google.com/uc?export=download&id=1doOe5vva88iEmPpy2gFWGDOHVlgwUrBc",
        },
        "sample":{
            "save_as":"sample_submission.csv",
            "url":"https://drive.google.com/uc?export=download&id=1LZeXbuFC_0Q8FPAgBFmW9a_YdKWXhGUx",
        }
    }

    def __init__(self):
        self._root = Path(__file__).parent.parent.parent

        # Logs initialized
        self._initialized_loggers = collections.defaultdict(bool)

        # Datasets promises
        self._train = None
        self._test = None
        self._sample = None

# ------------------ UTILS ------------------

    def remove_prefix(input_string: str, prefix: str) -> str:
        """Removes the prefix if exists at the beginning in the input string
        Needed for Python<3.9
        
        :param input_string: The input string
        :param prefix: The prefix
        :returns: The string without the prefix
        """
        if prefix and input_string.startswith(prefix):
            return input_string[len(prefix):]
        return input_string

    def as_relative(self, path: t.Union[str, Path]) -> Path:
        """Removes the prefix `self.root` from an absolute path.

        :param path: The absolute path
        :returns: A relative path starting at `self.root`
        """
        if type(path) == str:
            path = Path(path)
        return Path(CustomizedPath.remove_prefix(path.as_posix(), self.root.as_posix()))

    def mkdir_if_not_exists(self, path: Path, gitignore: bool=False) -> Path:
        """Makes the directory if it does not exists

        :param path: The input path
        :param gitignore: A boolean indicating if a gitignore must be included for the content of the directory
        :returns: The same path
        """
        path.mkdir(parents=True, exist_ok = True)

        if gitignore:
            with (path / '.gitignore').open('w') as f:
                f.write('*\n!.gitignore')

        return path

# ------------------ MAIN FOLDERS ------------------

    @property
    def root(self):
        return self._root

    @property
    def data(self):
        return self.mkdir_if_not_exists(self.root / 'data', gitignore=True)

    @property
    def output(self):
        return self.mkdir_if_not_exists(self.root / 'output', gitignore=True)

    @property
    def logs(self):
        return self.mkdir_if_not_exists(self.root / 'logs', gitignore=True)

# ------------------ DOWNLOADS ------------------

    def check_downloaded(
        self,
        name: str,
        *,
        force_reload: bool = False,
    ) -> None:
        """Checks that the file is already downloaded.

        :param name: The name of the asked file. Must be a key in self.files.
        :param force_reload: A boolean indicating if we must force re-downloading the file.
        """
        assert name in self.files, "Incorrect file name asked."

        target_path = project.data / self.files[name]['save_as']
        if not os.path.exists(target_path) or force_reload:
            response = requests.get(self.files[name]['url'], stream=True)
            with open(target_path, 'wb') as opt:
                opt.write(response.content)

    def data_as_pandas(
        self,
        name: str,
        *,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Returns a file as a pandas DataFrame.

        :param name: The name of the asked file. Must be a key in self.files.
        :param force_reload: A boolean indicating if we must force re-downloading the file.
        :returns: The pandas DataFrame corresponding to the asked file.
        """
        self.check_downloaded(name, force_reload = force_reload)
        return pd.read_csv(project.data / self.files[name]['save_as'], low_memory=False, sep=",")

    @property
    def train(self) -> pd.DataFrame:
        if self._train is None:
            self._train = self.data_as_pandas("train")
        return self._train

    @property
    def test(self) -> pd.DataFrame:
        if self._test is None:
            self._test = self.data_as_pandas("test")
        return self._test

    @property
    def sample(self) -> pd.DataFrame:
        if self._sample is None:
            self._sample = self.data_as_pandas("sample")

# ------------------ LOGS ------------------

    def get_log_file(self, logger_name: str) -> Path:
        """Creates and initializes a logger.

        :param logger_name: The logger name to create
        :returns: A path to the `logger_name.log` created and/or initialized file
        """
        file_name = logger_name + '.log'
        result = self.logs / file_name

        # Checking if exists
        if not os.path.isfile(result):
            with result.open('w') as f:
                pass

        # Header for new log
        if not self._initialized_loggers[logger_name]:
            with result.open('a') as f:
                f.write(f'\nNEW LOG AT {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')
            self._initialized_loggers[logger_name] = True

        return result

# ------------------ BERT CKECKPOINTS ------------------

    @property
    def bert_checkpoints(self):
        return self.mkdir_if_not_exists(self.output / 'bert_checkpoints')

    def get_new_bert_chepoint(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result = self.bert_checkpoints / ('checkpoint' + now)
        result.mkdir(parents=True, exist_ok = True)
        return result

    def get_latest_bert_checkpoint(self):
        directories = sorted([
            str(path)
            for path in self.bert_checkpoints.iterdir()
            if path.is_dir()
            and any(path.iterdir()) # Non-empty directory
        ])

        if len(directories) == 0:
            return None
        return Path(directories[-1]) / 'best'

# ------------------ FEATURES ------------------

    @property
    def features(self):
        return self.mkdir_if_not_exists(self.output / 'features')

    def get_feature_folder(self, feature_name):
        """Returns the feature folder for a given class of feature.
        Ex of class of feature: `'bert'`

        :param feature_name: The class of feature
        :returns: The path to the folder
        """
        return self.mkdir_if_not_exists(self.features / feature_name)

    def get_new_feature_file(self, feature_name: str, feature_type: str):
        """Returns a new feature file for a given class and a given type of feature.
        Ex of class of feature: `'bert'`
        Ex of type of feature: `'full_train'`

        :param feature_name: The class of feature
        :param feature_type: The type of feature
        :returns: The path to the new file
        """
        parent_folder = self.get_feature_folder(feature_name)
        result = parent_folder / f'{feature_type}_{feature_name}_{datetime.datetime.now().strftime("features_%Y_%m%d__%H_%M_%S")}.csv'
        with result.open('w') as f:
            pass
        return result

    def get_latest_features(self, feature_name: str, feature_type: str):
        """Returns the latest feature file for a given class and a given type of feature.
        Ex of class of feature: `'bert'`
        Ex of type of feature: `'full_train'`

        :param feature_name: The class of feature
        :param feature_type: The type of feature
        :returns: The path to the feature file
        """
        parent_folder = self.get_feature_folder(feature_name)
        files = sorted([
            str(path)
            for path in parent_folder.iterdir()
            if path.is_file()
            and feature_type in str(path)
        ])

        if len(files) == 0:
            return None
        return Path(files[-1])

# ------------------ XGBOOST ------------------

    @property
    def xgboost_folder(self):
        return self.mkdir_if_not_exists(self.output / 'xgboost')
    
    @property
    def xgboost_parameters(self):
        return self.xgboost_folder / 'params.json'

# ------------------ SUBMISSIONS ------------------

    @property
    def submission_folder(self):
        return self.mkdir_if_not_exists(self.output / 'submissions')
    
    def get_new_submission_file(self):
        """Returns a new submission file.

        :returns: The path to the new file
        """
        result = self.submission_folder / f'submission_{datetime.datetime.now().strftime("features_%Y_%m%d__%H_%M_%S")}.csv'
        with result.open('w') as f:
            pass
        return result

project = CustomizedPath() 