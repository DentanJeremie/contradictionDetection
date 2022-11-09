import os
from pathlib import Path
import typing as t

from matplotlib import pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split

from utils import data
from utils.data import Dataset
from utils.metrics import Evaluate
from utils.pathtools import project


class BertClassifier4Entailment(object):

    def __init__(
        self,
        sentences_1: t.List[str],
        sentences_2: t.List[str],
        labels: t.List[int],
        *,
        output_dir: Path = project.get_new_bert_chepoint(),
        model_name: str = 'bert-base-multilingual-cased',
        test_size: float = 0.03,
        checkpoint: t.Optional[Path] = None,
    ) -> None:
        """Initiates an instance of BertEncoder.

        TO BE COMPLETED
        """

        (
            self.train_sentences_1,
            self.test_sentences_1,
            self.train_sentences_2,
            self.test_sentences_2,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(
            sentences_1,
            sentences_2,
            labels,
            test_size = test_size
        )

        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast = True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels = 3)

        if checkpoint is not None:
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            print(f'Successfully loaded model from {checkpoint}')

        self.train_size = len(self.train_labels)
        self.test_size = len(self.test_labels)

        self._datasets_built = False
        self._trainer_built = False

    def tokenize(self):
        """
        TO BE COMPLETED
        """
        # Train set
        self.train_tokenized = self.tokenizer(
            self.train_sentences_1,
            self.train_sentences_2,
            padding=True,
            truncation=True,
            max_length=512,
        )
        self.train_dataset = Dataset(
            self.train_tokenized,
            self.train_labels,
        )

        # Test set
        self.test_tokenized = self.tokenizer(
            self.test_sentences_1,
            self.test_sentences_2,
            padding=True,
            truncation=True,
            max_length=512,
        )
        self.test_dataset = Dataset(
            self.test_tokenized,
            self.test_labels,
        )

        self._datasets_built = True

    def build_trainer(self):
        if not self._datasets_built:
            self.tokenize()

        self.args = TrainingArguments(
            # Output
            output_dir=self.output_dir,
            overwrite_output_dir = False,
            # Evaluation
            evaluation_strategy="epoch",
            # Saving
            save_strategy="epoch",
            save_total_limit=5,
            # Batches
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            # Training
            num_train_epochs=20,
            warmup_steps=100,
            # Best model
            load_best_model_at_end=True,
            metric_for_best_model='eval_accuracy',
            greater_is_better=True,
        )

        self.callback = EarlyStoppingCallback(
            early_stopping_patience=5
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=Evaluate.compute_metrics,
            callbacks=[self.callback]
        )

        self._trainer_built = True

    def train(self):
        if not self._trainer_built:
            self.build_trainer()

        self.trainer.train()
        self.trainer.save_model(self.output_dir / 'best')

    

if __name__ == '__main__':
    classifier = BertClassifier4Entailment(
        list(data.real.train.premise.values),
        list(data.real.train.hypothesis.values),
        list(data.real.train.label.values),
        checkpoint=project.get_newest_bert_checkpoint()
    )
    classifier.train()
