import csv
import logging
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import typing as t
import torch
from transformers import BertTokenizer, BertModel

from src.utils.constants import *
from src.utils.logging import logger
from src.utils.pathtools import project

logger.setLevel(logging.DEBUG)

class CosineSimilarityFeature():

    # ------------------ INIT ------------------

    def __init__(
        self,
        train_sentences_1: t.List[str],
        train_sentences_2: t.List[str],
        submission_sentences_1: t.List[str],
        submission_sentences_2: t.List[str],
        *,
        output_dir: Path = project.get_new_bert_chepoint(),
        model_name: str = 'bert-base-multilingual-cased'
    ) -> None:

        # Full train set
        self.train_sentences_1 = train_sentences_1
        self.train_sentences_2 = train_sentences_2
        self.train_size = len(self.train_sentences_1)

        # Submission test
        self.submission_sentences_1 = submission_sentences_1
        self.submission_sentences_2 = submission_sentences_2
        self.submission_size = len(self.submission_sentences_1)
        self.submission_size = len(self.submission_sentences_1)

        # Model
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, use_fast = True)
        self.model = BertModel.from_pretrained(self.model_name)

        self._tokens_built = False
        self._embeddings_built = False
        
# ------------------ TOKENIZATION ------------------

    def tokenize(self):
        """
        Tokenizes the train and submission sets.
        """
        # Train set
        logger.info('Tokenizing full train set...')
        self.train_sentences_1_tokenized = self.tokenizer(
            self.train_sentences_1,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=XML_MAX_LENGTH,
        )

        self.train_sentences_2_tokenized = self.tokenizer(
            self.train_sentences_2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=XML_MAX_LENGTH,
        )

        print(self.train_sentences_1_tokenized)
        print(self.train_sentences_2_tokenized)

        # Submission set
        logger.info('Tokenizing submission set...')

        self.submission_sentences_1_tokenized = self.tokenizer(
            self.submission_sentences_1,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=XML_MAX_LENGTH,
        )

        self.submission_sentences_2_tokenized = self.tokenizer(
            self.submission_sentences_2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=XML_MAX_LENGTH,
        )

        logger.info('Tokens built !')
        self._tokens_built = True

# ------------------ XML EMBEDDINGS ------------------

    def build_embeddings(self):

        if not self._tokens_built:
            self.tokenize()

        logger.info('Building XML embedding of the full train set...')

        ## Train set embedding 

        # Premise
        self.train_sentences_1_xml = self.model(**self.train_sentences_1_tokenized)
        self.train_sentences_1_xml_embedding = self.train_sentences_1_xml.last_hidden_state 
            # tensor of size  : number of sentences x number of tokens in one sentence (standardized with padding) x embedding size of one token

        # After we have produced our embeddings for each word in each sentence,
        # we need to perform a mean operation to create a single vector embedding
        # for each sequence.

        # For this mean operation, we need to take into account the attention_mask, 
        # in order to ignore padding tokens.


        self.train_1_mask = self.train_sentences_1_tokenized["attention_mask"].\
                                unsqueeze(-1).\
                                expand(self.train_sentences_1_xml_embedding.size()).\
                                float()

        self.train_1_xml_masked_embedding = self.train_1_mask * self.train_sentences_1_xml_embedding
        self.train_1_averaged_xml_masked_embedding = self.train_1_xml_masked_embedding.mean(dim=1)

        # Hypothesis
        self.train_sentences_2_xml = self.model(**self.train_sentences_2_tokenized) # give input_ids AND attention_mask, in order to throw the padding in embedding computing
        self.train_sentences_2_xml_embedding = self.train_sentences_2_xml.last_hidden_state

        self.train_2_mask = self.train_sentences_2_tokenized["attention_mask"].\
                                unsqueeze(-1).\
                                expand(self.train_sentences_2_xml_embedding.size()).\
                                float()

        self.train_2_xml_masked_embedding = self.train_2_mask * self.train_sentences_2_xml_embedding
        self.train_2_averaged_xml_masked_embedding = self.train_2_xml_masked_embedding.mean(dim=1)

        ## Submission set embedding

        logger.info('Building XML embedding of the submission set...')

        # Premise
        self.submission_sentences_1_xml = self.model(**self.submission_sentences_1_tokenized)
        self.submission_sentences_1_xml_embedding = self.submission_sentences_1_xml.last_hidden_state

        self.submission_1_mask = self.submission_sentences_1_tokenized["attention_mask"].\
                                unsqueeze(-1).\
                                expand(self.submission_sentences_1_xml_embedding.size()).\
                                float()

        self.submission_1_xml_masked_embedding = self.submission_1_mask * self.submission_sentences_1_xml_embedding
        self.submission_1_averaged_xml_masked_embedding = self.submission_1_xml_masked_embedding.mean(dim=1)

        # Hypothesis
        self.submission_sentences_2_xml = self.model(**self.submission_sentences_2_tokenized)
        self.submission_sentences_2_xml_embedding = self.submission_sentences_2_xml.last_hidden_state

        self.submission_2_mask = self.submission_sentences_2_tokenized["attention_mask"].\
                                unsqueeze(-1).\
                                expand(self.submission_sentences_2_xml_embedding.size()).\
                                float()

        self.submission_2_xml_masked_embedding = self.submission_2_mask * self.submission_sentences_2_xml_embedding
        self.submission_2_averaged_xml_masked_embedding = self.submission_2_xml_masked_embedding.mean(dim=1)

        logger.info('XML embeddings built!')
        self._embeddings_built = True

# ------------------ COSINE SIMILARITY ------------------

    def cosine_sim(self):

        if not self._embeddings_built:
            self.build_embeddings()

        logger.info('Computing cosine similarity feature for the full train set...')

        # Convert from PyTorch tensor to numpy array
        self.train_1_averaged_xml_masked_embedding = self.train_1_averaged_xml_masked_embedding.detach().numpy()
        self.train_2_averaged_xml_masked_embedding = self.train_2_averaged_xml_masked_embedding.detach().numpy()

        print(self.train_1_averaged_xml_masked_embedding)
        print(self.train_2_averaged_xml_masked_embedding)

        self.submission_1_averaged_xml_masked_embedding = self.submission_1_averaged_xml_masked_embedding.detach().numpy()
        self.submission_2_averaged_xml_masked_embedding = self.submission_2_averaged_xml_masked_embedding.detach().numpy()
        
        with open(project.get_new_feature_file(COSINE_FEATURE_NAME, FULL_TRAIN_FEAETURE_TYPE), 'w') as f:
            csv_out = csv.writer(f)
            csv_out.writerow([DATA_ID, COSINE_FEATURE_NAME])
            for id in range(self.train_size):
                sim = cosine_similarity([self.train_1_averaged_xml_masked_embedding[id,:]], [self.train_2_averaged_xml_masked_embedding[id,:]])
                csv_out.writerow([id, sim[0][0]])

        logger.info('Computing cosine similarity feature for the submission set...')

        with open(project.get_new_feature_file(COSINE_FEATURE_NAME, SUBMISSION_FEAETURE_TYPE), 'w') as f:
            csv_out = csv.writer(f)
            csv_out.writerow([DATA_ID, COSINE_FEATURE_NAME])
            for id in range(self.submission_size):
                sim = cosine_similarity([self.submission_1_averaged_xml_masked_embedding[id,:]], [self.submission_2_averaged_xml_masked_embedding[id,:]])
                csv_out.writerow([id, sim[0][0]])

        logger.info('Cosine similarity features computation: done!')


def main():
    

    cosine_embedding = CosineSimilarityFeature(
        train_sentences_1=list(project.train[DATA_PREMISE].values),
        train_sentences_2=list(project.train[DATA_HYPOTHESIS].values),
        submission_sentences_1=list(project.test[DATA_PREMISE].values),
        submission_sentences_2=list(project.test[DATA_HYPOTHESIS].values),
    )
    cosine_embedding.cosine_sim()


if __name__ == '__main__':
    main()
