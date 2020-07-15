""" A module containing the models to be trained on gene expression data """

import pickle
from abc import ABC, abstractmethod
from typing import Union, Iterable

import numpy as np
import sklearn.linear_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from neptune.experiments import Experiment

from saged.datasets import LabeledDataset, UnlabeledDataset


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
    """
    A model API similar to the scikit-learn API that will specify the
    base acceptable functions for models in this module's benchmarking code
    """

    def __init__(self) -> None:
        """
        Standard model init function. We use pass instead of raising a NotImplementedError
        here in case inheriting classes decide to call `super()`
        """
        pass

    @abstractmethod
    def load_model(classobject, model_path):
        """
        Read a pickeled model from a file and return it

        Arguments
        ---------
        model_path: The location where the model is stored

        Returns
        -------
        model: The model saved at `model_path`
        """
        raise NotImplementedError

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
    def predict(self, dataset: UnlabeledDataset) -> np.ndarray:
        """
        Predict the labels for a

        Arguments
        ---------
        dataset: The unlabeled data whose labels should be predicted

        Returns
        -------
        predictions: A numpy array of predictions
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, out_path: str) -> None:
        """
        Write the model to a file

        Arguments
        ---------
        out_path: The path to the file to write the classifier to

        Raises
        ------
        FileNotFoundError if out_path isn't openable
        """
        raise NotImplementedError


class LogisticRegression(ExpressionModel):
    """ A model API similar to the scikit-learn API that will specify the
    base acceptable functions for models in this module's benchmarking code
    """

    def __init__(self, seed: int = 42) -> None:
        """
        The initializer the LogisticRegression class

        seed: The random seed to be used by the model
        """
        self.model = sklearn.linear_model.LogisticRegression(random_state=seed)

    def fit(self, dataset: LabeledDataset) -> "LogisticRegression":
        """
        Train a model using the given labeled data

        Arguments
        ---------
        dataset: The labeled data for use in training

        Returns
        -------
        self: The fitted model
        """
        X, y = dataset.get_all_data()

        self.model = self.model.fit(X, y)
        return self

    def predict(self, dataset: UnlabeledDataset) -> np.ndarray:
        """
        Use the model to predict the labels for a given unlabeled dataset

        Arguments
        ---------
        dataset: The unlabeled data whose labels should be predicted

        Returns
        -------
        predictions: A numpy array of predictions
        """
        X = dataset.get_all_data()
        return self.model.predict(X)

    def save_model(self, out_path: str) -> None:
        """
        Write the classifier to a file

        Arguments
        ---------
        out_path: The path to the file to write the classifier to

        Raises
        ------
        FileNotFoundError if out_path isn't openable
        """

        with open(out_path, 'wb') as out_file:
            pickle.dump(self, out_file)

    @classmethod
    def load_model(classobject, model_path):
        """
        Read a pickeled model from a file and return it

        Arguments
        ---------
        model_path: The location where the model is stored

        Returns
        -------
        model: The model saved at `model_path`
        """
        with open(model_path, 'rb') as model_file:
            return pickle.load(model_file)


class ThreeLayerClassifier(nn.Module):
    """ A basic three layer neural net for use in wrappers like FullyConnectedNet """
    def __init__(self, model_params: dict):
        input_size = model_params['input_size']

        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        self.fc3 = nn.Linear(input_size // 4, 1)

    def forward(self, x: torch.Tenor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).view(-1)

        return x


class SupervisedNet(ExpressionModel):
    """
    A wrapper class implementing the ExpressionModel API while remaining modular enough
    to accept any supervised classifier implementing the nn.Module API
    """
    def __init__(self,
                 experiment: Experiment,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn._WeightedLoss,
                 model_class: nn.Module) -> None:
        """
        Standard model init function. We use pass instead of raising a NotImplementedError
        here in case inheriting classes decide to call `super()`

        Arguments
        ---------
        experiment: The neptune experiment for the model. Enables logging and tracks
            hyperparameter information
        optimizer: The optimizer to be used when training the model
        loss_fn: The loss function class to use
        model_class: The type of classifier to use
        """
        experiment.set_property('model', str(type(model_class)))
        self.experiment = experiment
        model_params = experiment.get_properties()

        self.model = model_class(model_params)
        lr = model_params['lr']
        weight_decay = model_params.get('weight_decay', 0)
        self.optimizer = optimizer(self.model.get_parameters(),
                                   lr=lr,
                                   weight_decay=weight_decay)

        # Load the weight for each class if the loss function is a weighted loss
        if torch.nn._WeightedLoss in loss_fn.__bases__:
            weight = model_params.get('weight', None)
            self.loss_fn = loss_fn(weight=weight)
        else:
            self.loss_fn = loss_fn()

    @classmethod
    def load_model(classobject,
                   checkpoint_path: str,
                   experiment: Experiment,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: torch.nn._WeightedLoss,
                   model_class: nn.Module) -> "SupervisedNet":
        """
        Read a pickeled model from a file and return it

        Arguments
        ---------
        checkpoint_path: The location where the model is stored
        experiment: The neptune experiment for the model. Enables logging and tracks
            hyperparameter information
        optimizer: The optimizer to be used when training the model
        loss_fn: The loss function class to use
        model_class: The type of classifier to use

        Returns
        -------
        model: The loaded model
        """
        model = classobject(experiment,
                            optimizer,
                            loss_fn,
                            model_class)

        state_dicts = torch.load(checkpoint_path)
        model.load_parameters(state_dicts['model_state_dict'])
        model.optimizer.load_state_dict(state_dicts['optimizer_state_dict'])

        return model

    @abstractmethod
    def save_model(self, out_path: str) -> None:
        """
        Write the model to a file

        Arguments
        ---------
        out_path: The path to the file to write the classifier to

        Raises
        ------
        FileNotFoundError if out_path isn't openable
        """
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dicT(),
                    },
                   out_path
                   )

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
    def predict(self, dataset: UnlabeledDataset) -> np.ndarray:
        """
        Predict the labels for a

        Arguments
        ---------
        dataset: The unlabeled data whose labels should be predicted

        Returns
        -------
        predictions: A numpy array of predictions
        """
        raise NotImplementedError

    def get_parameters(self) -> Iterable[torch.Tensor]:
        return self.model.state_dict()

    def load_parameters(self, parameters: dict):
        self.model.load_state_dict(parameters)
