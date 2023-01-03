# Data facts
NUM_LABELS = 3
TEST_SIZE = 0.03
DATA_PREMISE = 'premise'
DATA_HYPOTHESIS = 'hypothesis'
DATA_LABEL = 'label'
DATA_ID = 'id'

# Features
SUBMISSION_FEAETURE_TYPE = 'submission'
FULL_TRAIN_FEAETURE_TYPE = 'full_train'
FEATURE_NAMES = ['bert', 'cosine']
BERT_FEATURE_NAME = 'bert'
COSINE_FEATURE_NAME = 'cosine'

# BERT config
BERT_EVALUATION_STRATEGY = "epoch"
BERT_GRADIENT_ACCUMULATION_STEPS = 1
BERT_GREATER_IS_BETTER = True
BERT_MAX_LENGTH = 512
BERT_METRIC_FOR_BEST_MODEL = 'eval_accuracy'
BERT_NUM_TRAIN_EPOCHS = 0.1
BERT_PER_DEVICE_EVAL_BATCH_SIZE = 8
BERT_PER_DEVICE_TRAIN_BATCH_SIZE = 8
BERT_SAVE_STRATEGY = "epoch"
BERT_TOTAL_SAVE_LIMIT = 3
BERT_WARMUM_STEPS = 100

# XML CONFIG
XML_MAX_LENGTH = 512

# XGBOOST
XGB_PARAM_SEARCH = {
    'max_depth': [2],
    'learning_rate': [0.01],
    'colsample_bytree': [0.3, 0.8]
}
XGB_DEFAULT_PARAM_TO_SEARCH = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'colsample_bytree': 0.8,
}
XGB_ADDITIONNAL_PARAM = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'min_child_weight': 2,
    'eta': 0.3,
    'subsample': 0.5,
    'gamma': 1,
    'eval_metric': 'error',
}
