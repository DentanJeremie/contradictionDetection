import typing as t

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch

from src.utils.logging import logger


class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

    @classmethod
    def compute_metrics(
        cls,
        p: t.Tuple[torch.Tensor, torch.Tensor]
    ) -> t.Dict[str, float]:
        """Compute the main accuracy metrics of the inputs.

        :param p: A tuple `(predictions:torch.Tensor, labels:torch.Tensor)`
        :returns: A dict{metric->value} where `metric in ['accuracy', 'precision', 'recall', 'f1']`
        """
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average='weighted', zero_division=0.0)
        precision = precision_score(y_true=labels, y_pred=pred, average='weighted', zero_division=0.0)
        f1 = f1_score(y_true=labels, y_pred=pred, average='weighted', zero_division=0.0)

        result = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        return result