""" A module containing the models to be trained on gene expression data """

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from datasets import LabeledDataset


class ModelResults():
    """
    A structure for storing the metrics corresponding to a model's training.
    The ModelResults class requires a loss and another metric such as accuracy or F1 score.
    Other metrics can be tracked via the `add_other` method
    """
    def __init__(self,
                 model_name: str,
                 progress_type: str,
                 loss_type: str,
                 metric_type: str
                 ) -> None:
        """
        Initialize the ModelResults object and keep track of its loss and other metrics.

        Arguments
        ---------
        model_name: The name of the model that produced these results
        progress_type: The unit of the values that will be stored in the progress array
        loss_type: The name of the loss function being used
        metric_type: The name of the metric being used
        """
        self.name = model_name
        self.loss_type = loss_type
        self.metric_type = metric_type
        self.progress_type = progress_type

        # Progress stores the number of iterations of type progress_type so far
        self.val_progress = []
        self.train_progress = []
        self.val_loss = []
        self.train_loss = []
        self.val_metric = []
        self.train_metric = []
        self.other = {}

    def add_progress(self,
                     progress: int,
                     loss: float,
                     metric: float,
                     is_val: bool
                     ) -> None:
        """
        Update the ModelResults with loss and metric information

        Arguments
        ---------
        progress: The step, epoch, etc. that this entry corresponds to
        loss: The value of the loss function at this time
        metric: The value of the metric at this time
        is_val: Whether the results should be stored as validation metrics or training metrics
        """
        if is_val:
            self.val_progress.append(progress)
            self.val_loss.append(loss)
            self.val_metric.append(metric)
        else:
            self.train_progress.append(progress)
            self.train_loss.append(loss)
            self.train_metric.append(metric)

    def add_other(self,
                  metric_name: str,
                  metric_val: Union[int, float],
                  metric_progress: int
                  ) -> None:
        """
        Add information about an additional metric to the model

        Arguments
        ---------
        metric_name: The name of the metric being recorded
        metric_val: The value of the metric being recorded
        metric_progress: The step, epoch, etc. that this entry corresponds to
        """

        if metric_name in self.other:
            self.other['metric_name']['vals'].append(metric_val)
            self.other['metric_name']['progress'].append(metric_progress)

        else:
            self.other['metric_name'] = {'vals': [metric_val],
                                         'progress': [metric_progress]
                                         }


class ExpressionModel(ABC):
    """ A model API similar to the scikit-learn API that will specify the
    base acceptable functions for models in this module's benchmarking code
    """

    def __init__(self) -> None:
        """
        Standard model init function. We use pass instead of raising a NotImplementedError
        here in case inheriting classes decide to call `super()`
        """
        pass

    @abstractmethod
    def fit(self, dataset: LabeledDataset) -> ModelResults:
        """
        Train a model using the given labeled data

        Arguments
        ---------
        dataset: The labeled data for use in training

        Returns
        -------
        results: The metrics produced during the training process
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset: np.array) -> np.array:
        """

        Arguments
        ---------
        dataset: The unlabeled data whose labels should be predicted

        Returns
        -------
        predictions: A numpy array of predictions
        """
        raise NotImplementedError
