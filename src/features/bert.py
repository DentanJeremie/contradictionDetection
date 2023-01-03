import csv
import logging
import os
from pathlib import Path
import typing as t

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from scipy.special import softmax
from sklearn.model_selection import train_test_split

from src.utils.constants import *
from src.utils.datasets import Dataset
from src.utils.logging import logger
from src.utils.pathtools import project

logger.setLevel(logging.DEBUG)

# Setting up tokenizer parallism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BertClassifier4Entailment(object):

# ------------------ INIT ------------------

    def __init__(
        self,
        train_sentences_1: t.List[str],
        train_sentences_2: t.List[str],
        train_labels: t.List[int],
        submission_sentences_1: t.List[str],
        submission_sentences_2: t.List[str],
        *,
        output_dir: Path = project.get_new_bert_chepoint(),
        model_name: str = 'bert-base-multilingual-cased',
        test_size: float = TEST_SIZE,
        checkpoint: t.Optional[Path] = None,
    ) -> None:
        """Initiates an instance of BertEncoder.

        TO BE COMPLETED
        """

        logger.debug('Initiating a BERT classifier...')

        # Full train set
        #   List of premise sentences in full train set
        self.full_train_sentences_1 = train_sentences_1
        #   List of hypothesis sentences in full train set
        self.full_train_sentences_2 = train_sentences_2
        self.full_train_labels = train_labels

        # Partial train and test sets
        (
            self.train_sentences_1,
            self.test_sentences_1,
            self.train_sentences_2,
            self.test_sentences_2,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(
            train_sentences_1,
            train_sentences_2,
            train_labels,
            test_size = test_size
        )

        # Submission set
        self.submission_sentences_1 = submission_sentences_1
        self.submission_sentences_2 = submission_sentences_2

        # Models
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast = True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels = NUM_LABELS
        )

        if checkpoint is not None:
            logger.info('Found an existing trained model')
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            logger.info(f'Successfully loaded model from {checkpoint}')
        else:
            logger.info('No existing trained model specified')

        self.train_size = len(self.train_labels)
        self.test_size = len(self.test_labels)

        self._datasets_built = False
        self._trainer_built = False

# ------------------ TOKENIZATION ------------------

    def tokenize(self):
        """
        Tokenizes the train, test and submission sets.
        The following variables are initiated:

        * `self.train_dataset: Dataset`
        * `self.test_dataset: Dataset`
        * `self.submission_tokenized: Dataset`
        * `self.full_train_dataset: Dataset`
        """
        # Train set
        logger.info('Tokenizing train set...')
        self.train_tokenized = self.tokenizer(
            self.train_sentences_1,
            self.train_sentences_2,
            padding=True,
            truncation=True,
            max_length=BERT_MAX_LENGTH,
        )
        self.train_dataset = Dataset(
            self.train_tokenized,
            self.train_labels,
        )

        # Test set
        logger.info('Tokenizing test set...')
        self.test_tokenized = self.tokenizer(
            self.test_sentences_1,
            self.test_sentences_2,
            padding=True,
            truncation=True,
            max_length=BERT_MAX_LENGTH,
        )
        self.test_dataset = Dataset(
            self.test_tokenized,
            self.test_labels,
        )

        # Test set
        logger.info('Tokenizing submission set...')
        self.submission_tokenized = self.tokenizer(
            self.submission_sentences_1,
            self.submission_sentences_2,
            padding=True,
            truncation=True,
            max_length=BERT_MAX_LENGTH,
        )
        self.submission_dataset = Dataset(
            self.submission_tokenized,
            # No label
        )

        # Full train set
        logger.info('Tokenizing full train set...')
        self.full_train_tokenized = self.tokenizer(
            self.full_train_sentences_1,
            self.full_train_sentences_2,
            padding=True,
            truncation=True,
            max_length=BERT_MAX_LENGTH,
        )
        self.full_train_dataset = Dataset(
            self.full_train_tokenized,
            self.full_train_labels,
        )

        logger.info('Datasets built !')
        self._datasets_built = True

# ------------------ BUILDING TRAINER ------------------

    def build_trainer(self):
        """
        Builds the BERT trainer.
        The hyperparameters are defines in `src.utils.constants`
        """
        if not self._datasets_built:
            self.tokenize()

        logger.info('Building BERT trainer...')

        self.args = TrainingArguments(
            # Output
            output_dir=self.output_dir,
            overwrite_output_dir = False,
            # Evaluation
            evaluation_strategy=BERT_EVALUATION_STRATEGY,
            # Saving
            save_strategy=BERT_SAVE_STRATEGY,
            save_total_limit=BERT_TOTAL_SAVE_LIMIT,
            # Batches
            per_device_train_batch_size=BERT_PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=BERT_PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=BERT_GRADIENT_ACCUMULATION_STEPS,
            # Training
            num_train_epochs=BERT_NUM_TRAIN_EPOCHS,
            warmup_steps=BERT_WARMUM_STEPS,
            # Best model
            load_best_model_at_end=True,
            metric_for_best_model=BERT_METRIC_FOR_BEST_MODEL,
            greater_is_better=BERT_GREATER_IS_BETTER,
        )

        self.callback = EarlyStoppingCallback(
            early_stopping_patience=5
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=Dataset.compute_metrics,
            callbacks=[self.callback]
        )

        logger.info('Trainer built !')
        self._trainer_built = True

# ------------------ TRAIN ------------------

    def train_logging(self):
        epoch_metrics = {}
        for info_dict in self.trainer.state.log_history:
            if 'eval_accuracy' in info_dict:
                epoch_metrics[info_dict['epoch']] = {
                    'eval_loss': info_dict['eval_loss'],
                    'eval_accuracy': info_dict['eval_accuracy'],
                    'eval_precision': info_dict['eval_precision'],
                    'eval_recall': info_dict['eval_recall'],
                    'eval_f1': info_dict['eval_f1'],
                }
            if 'train_runtime' in info_dict:
                train_metrics = {
                    'train_runtime': info_dict['train_runtime'],
                    'train_samples_per_second': info_dict['train_samples_per_second'],
                }
        for epoch in sorted(epoch_metrics):
            for metric in sorted(epoch_metrics[epoch]):
                logger.info(f'Epoch {epoch}: {metric}: {epoch_metrics[epoch][metric]:.3f}')
        for metric in sorted(train_metrics):
            logger.info(f'Training metrics: {metric}: {train_metrics[metric]:.3f}')

    def train(self):
        if not self._trainer_built:
            self.build_trainer()

        logger.info(f'Starting training of the BERT classifier ({BERT_NUM_TRAIN_EPOCHS} epochs)')
        self.trainer.train()
        self.train_logging()
        logger.info('Training: done !')
        self.trainer.save_model(self.output_dir / 'best')
        logger.info(f'Best model stored at {self.output_dir / "best"}')

# ------------------ PREDICT ------------------

    def predict_logging(self, full_train_y, full_train_labels):
        prediction_metrics = Dataset.compute_metrics((full_train_y, full_train_labels))
        for metric in prediction_metrics:
            logger.info(f'Full training preds.: {metric}: {prediction_metrics[metric]:.3f}')
        
    def predict(self):
        if not self._datasets_built:
            self.tokenize()

        logger.info('Predicting features for the full train set...')
        full_train_raw_pred, _, _ = self.trainer.predict(self.full_train_dataset)
        # array of size (n_full_train,3) : each column gives the probability of the train sample 
        # to belong to class 0, 1 or 2
        full_train_y = softmax(full_train_raw_pred, axis=1) 
        with open(project.get_new_feature_file(BERT_FEATURE_NAME, FULL_TRAIN_FEAETURE_TYPE), 'w') as f:
            csv_out = csv.writer(f)
            csv_out.writerow([DATA_ID, BERT_FEATURE_NAME])
            for id, row in enumerate(full_train_y):
                csv_out.writerow([id, row[0], row[1], row[2]])

        self.predict_logging(full_train_y, self.full_train_labels)

        logger.info('Predicting features for the submission set...')
        submission_raw_pred, _, _ = self.trainer.predict(self.submission_dataset)
        submission_y = softmax(submission_raw_pred, axis=1)
        with open(project.get_new_feature_file(BERT_FEATURE_NAME, SUBMISSION_FEAETURE_TYPE), 'w') as f:
            csv_out = csv.writer(f)
            csv_out.writerow([DATA_ID, BERT_FEATURE_NAME])
            for id, row in enumerate(submission_y):
                csv_out.writerow([id, row[0], row[1], row[2]])

        logger.info('Predictions: done!')

def main():
    classifier = BertClassifier4Entailment(
        train_sentences_1=list(project.train[DATA_PREMISE].values),
        train_sentences_2=list(project.train[DATA_HYPOTHESIS].values),
        train_labels=list(project.train[DATA_LABEL].values),
        submission_sentences_1=list(project.test[DATA_PREMISE].values),
        submission_sentences_2=list(project.test[DATA_HYPOTHESIS].values),
        checkpoint=project.get_latest_bert_checkpoint(),
    )
    classifier.train()
    classifier.predict()

if __name__ == '__main__':
    main()
