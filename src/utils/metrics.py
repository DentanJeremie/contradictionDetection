import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class Evaluate(object):

    def compute_metrics(p):
        """
        TO BE COMPLETED
        """
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
        precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
        f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}