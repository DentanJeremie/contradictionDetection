import logging
import os

import numpy as np
import pandas as pd
import requests

from utils.pathtools import project

class Downloader():
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

    # Private attributes
    _train = None
    _test = None
    _sample = None

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
            logging.info(f"Downloading {self.files[name]['save_as']} from {self.files[name]['url']} ...")
            response = requests.get(self.files[name]['url'], stream=True)
            with open(target_path, 'wb') as opt:
                opt.write(response.content)

    def as_pandas(
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
        return pd.read_csv(project.data / self.files[name]['save_as'], low_memory=False)

    @property
    def train(self) -> pd.DataFrame:
        if self._train is None:
            self._train = self.as_pandas("train")
        return self._train

    @property
    def test(self) -> pd.DataFrame:
        if self._test is None:
            self._test = self.as_pandas("test")
        return self._test

    @property
    def sample(self) -> pd.DataFrame:
        if self._sample is None:
            self._sample = self.as_pandas("sample")
        return self._sample

real = Downloader()