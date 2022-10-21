from transformers import BertTokenizer, TFBertModel
from matplotlib import pyplot as plt
import tensorflow as tf

model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

