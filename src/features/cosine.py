import csv
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import typing as t
import torch

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
        train_labels: t.List[int],
        submission_sentences_1: t.List[str],
        submission_sentences_2: t.List[str],
        *,
        output_dir: Path = project.get_new_bert_chepoint(),
        model_name: str = 'paraphrase-multilingual-mpnet-base-v2',
        model_nickname: str = ''
    ) -> None:

        # Full train set
        self.train_sentences_1 = train_sentences_1
        self.train_sentences_2 = train_sentences_2
        self.train_labels = train_labels
        self.train_size = len(self.train_sentences_1)

        # Submission set
        self.submission_sentences_1 = submission_sentences_1
        self.submission_sentences_2 = submission_sentences_2

        # Submission test
        self.submission_sentences_1 = submission_sentences_1
        self.submission_sentences_2 = submission_sentences_2
        self.submission_size = len(self.submission_sentences_1)
        self.submission_size = len(self.submission_sentences_1)

        # Model
        self.model_name = model_name
        self.model_nickname = model_nickname
        self.output_dir = output_dir
        self.model = SentenceTransformer(self.model_name)

        self._embedding_built = False

# ------------------ VECTOR EMBEDDING ------------------

    def compute_embedding(self):

        logger.info('Computing embedding for the model :' + self.model_nickname)

        self.train_sentences_1_embedding = self.model.encode(self.train_sentences_1)
        self.train_sentences_2_embedding = self.model.encode(self.train_sentences_2)

        self.submission_sentences_1_embedding = self.model.encode(self.submission_sentences_1)
        self.submission_sentences_2_embedding = self.model.encode(self.submission_sentences_2)

        self._embedding_built = True
        
# ------------------ COSINE SIMILARITY ------------------

    def cosine_sim(self):

        if not self._embedding_built:
            self.compute_embedding()

        logger.info(f'Computing cosine similarity feature for the full train set under model {self.model_nickname}...')

        with open(project.get_new_feature_file(COSINE_FEATURE_NAME + "-" + self.model_nickname, FULL_TRAIN_FEAETURE_TYPE), 'w') as f:
            csv_out = csv.writer(f, lineterminator='\n')
            csv_out.writerow([DATA_ID, COSINE_FEATURE_NAME + "-" + self.model_nickname])
            for id in range(self.train_size):
                sim = cosine_similarity([self.train_sentences_1_embedding[id,:]], [self.train_sentences_2_embedding[id,:]])
                csv_out.writerow([id, sim[0][0]])

        logger.info(f'Computing cosine similarity feature for the submission set under model {self.model_nickname}...')

        with open(project.get_new_feature_file(COSINE_FEATURE_NAME + "-" + self.model_nickname, SUBMISSION_FEAETURE_TYPE), 'w') as f:
            csv_out = csv.writer(f, lineterminator='\n')
            csv_out.writerow([DATA_ID, COSINE_FEATURE_NAME + "-" + self.model_nickname])
            for id in range(self.submission_size):
                sim = cosine_similarity([self.submission_sentences_1_embedding[id,:]], [self.submission_sentences_2_embedding[id,:]])
                csv_out.writerow([id, sim[0][0]])

        logger.info('Cosine similarity features computation: done!')

# ------------------ ANTONYM DISTANCE SIMILARITY ------------------

    def antonym_sim(self):

        if not self._embedding_built:
            self.compute_embedding()

        self.antonym_distance = abs(self.train_sentences_1_embedding - self.train_sentences_2_embedding) # number of sequences x embedding size

        self.avg_antonym_distance_entailment = np.mean(self.antonym_distance[np.where(np.array(self.train_labels) == 0)[0]], axis = 0)
        self.avg_antonym_distance_neutral = np.mean(self.antonym_distance[np.where(np.array(self.train_labels) == 1)[0]], axis = 0)
        self.avg_antonym_distance_contradiction = np.mean(self.antonym_distance[np.where(np.array(self.train_labels) == 2)[0]], axis = 0)

        logger.info(f'Computing antonym similarity feature for the full train set under model {self.model_nickname}...')

        self.train_antonym_similarity = cosine_similarity(
                                    np.array([self.avg_antonym_distance_entailment, 
                                    self.avg_antonym_distance_neutral, 
                                    self.avg_antonym_distance_contradiction]), 
                                    abs(self.train_sentences_1_embedding - self.train_sentences_2_embedding)
                                    ).T

        
        self.train_antonym_similarity = pd.DataFrame(data = self.train_antonym_similarity,
                                columns = [ANTONYM_FEATURE_NAME + "-" + self.model_nickname + "-0", 
                                                    ANTONYM_FEATURE_NAME + "-" + self.model_nickname + "-1", 
                                                    ANTONYM_FEATURE_NAME + "-" + self.model_nickname + "-2"])

        self.train_antonym_similarity.index.name = DATA_ID

        self.train_antonym_similarity.to_csv(project.get_new_feature_file(ANTONYM_FEATURE_NAME + "-" + self.model_nickname, FULL_TRAIN_FEAETURE_TYPE))

        logger.info(f'Computing antonym similarity feature for the submission set under model {self.model_nickname}...')

        self.submission_antonym_similarity = cosine_similarity(
                                    np.array([self.avg_antonym_distance_entailment, 
                                    self.avg_antonym_distance_neutral, 
                                    self.avg_antonym_distance_contradiction]), 
                                    abs(self.submission_sentences_1_embedding - self.submission_sentences_2_embedding)
                                    ).T
        
        self.submission_antonym_similarity = pd.DataFrame(data = self.submission_antonym_similarity,
                                columns = [ANTONYM_FEATURE_NAME + "-" + self.model_nickname + "-0", 
                                                    ANTONYM_FEATURE_NAME + "-" + self.model_nickname + "-1", 
                                                    ANTONYM_FEATURE_NAME + "-" + self.model_nickname + "-2"])

        self.submission_antonym_similarity.index.name = DATA_ID

        self.submission_antonym_similarity.to_csv(project.get_new_feature_file(ANTONYM_FEATURE_NAME + "-" + self.model_nickname, SUBMISSION_FEAETURE_TYPE))
     
        logger.info('Computing antonym similarity features computation: done!')

def main():

    cosine_sim_distiluse = CosineSimilarityFeature(
        train_sentences_1=list(project.train[DATA_PREMISE].values),
        train_sentences_2=list(project.train[DATA_HYPOTHESIS].values),
        train_labels=list(project.train[DATA_LABEL].values),
        submission_sentences_1=list(project.test[DATA_PREMISE].values),
        submission_sentences_2=list(project.test[DATA_HYPOTHESIS].values),
        model_name='distiluse-base-multilingual-cased-v2',
        model_nickname=DISTILUSE_SUBFEATURE_NAME
        )

    cosine_sim_distiluse.cosine_sim()
    cosine_sim_distiluse.antonym_sim()

    cosine_sim_minilm = CosineSimilarityFeature(
        train_sentences_1=list(project.train[DATA_PREMISE].values),
        train_sentences_2=list(project.train[DATA_HYPOTHESIS].values),
        train_labels=list(project.train[DATA_LABEL].values),
        submission_sentences_1=list(project.test[DATA_PREMISE].values),
        submission_sentences_2=list(project.test[DATA_HYPOTHESIS].values),
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        model_nickname=MINILM_SUBFEATURE_NAME
        )
    cosine_sim_minilm.cosine_sim()
    cosine_sim_minilm.antonym_sim()

    cosine_sim_mpnet = CosineSimilarityFeature(
        train_sentences_1=list(project.train[DATA_PREMISE].values),
        train_sentences_2=list(project.train[DATA_HYPOTHESIS].values),
        train_labels=list(project.train[DATA_LABEL].values),
        submission_sentences_1=list(project.test[DATA_PREMISE].values),
        submission_sentences_2=list(project.test[DATA_HYPOTHESIS].values),
        model_name='paraphrase-multilingual-mpnet-base-v2',
        model_nickname=MPNET_SUBFEATURE_NAME
        )
    cosine_sim_mpnet.cosine_sim()
    cosine_sim_mpnet.antonym_sim()


if __name__ == '__main__':
    main()
