""" A module containing the models to be trained on gene expression data """

import copy
import pickle
from abc import ABC, abstractmethod
from typing import Union, Iterable

import neptune
import numpy as np
import sklearn.linear_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import saged.utils as utils
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
        super(ThreeLayerClassifier, self).__init__()
        input_size = model_params['input_size']
        output_size = model_params['output_size']

        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        self.fc3 = nn.Linear(input_size // 4, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class PytorchSupervised(ExpressionModel):
    """
    A wrapper class implementing the ExpressionModel API while remaining modular enough
    to accept any supervised classifier implementing the nn.Module API
    """
    def __init__(self,
                 config: dict,
                 optimizer: torch.optim.Optimizer,
                 loss_class: torch.nn.modules.loss._WeightedLoss,
                 model_class: nn.Module) -> None:
        """
        Standard model init function for a supervised model

        Arguments
        ---------
        config: The configuration information for the model
        optimizer: The optimizer to be used when training the model
        loss_class: The loss function class to use
        model_class: The type of classifier to use
        """
        # TODO create custom config parsing function to document the parameters
        model_config = config['model']
        self.config = config
        self.model = model_class(model_config)
        lr = float(model_config['lr'])
        weight_decay = float(model_config.get('weight_decay', 0))
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=lr,
                                   weight_decay=weight_decay)

        self.loss_class = loss_class

        self.device = torch.device(model_config.get('device', 'cpu'))

        torch.manual_seed = model_config['seed']
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def load_model(classobject,
                   checkpoint_path: str,
                   config: dict,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: torch.nn.modules.loss._WeightedLoss,
                   model_class: nn.Module) -> "PytorchSupervised":
        """
        Read a pickled model from a file and return it

        Arguments
        ---------
        checkpoint_path: The location where the model is stored
        config: The configuration information for the model
        optimizer: The optimizer to be used when training the model
        loss_fn: The loss function class to use
        model_class: The type of classifier to use

        Returns
        -------
        model: The loaded model
        """
        model = classobject(config,
                            optimizer,
                            loss_fn,
                            model_class)

        state_dicts = torch.load(checkpoint_path)
        model.load_parameters(state_dicts['model_state_dict'])
        model.optimizer.load_state_dict(state_dicts['optimizer_state_dict'])

        return model

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
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    },
                   out_path
                   )

    def fit(self, dataset: LabeledDataset) -> "PytorchSupervised":
        """
        Train a model using the given labeled data

        Arguments
        ---------
        dataset: The labeled data for use in training

        Returns
        -------
        results: The metrics produced during the training process
        """
        # Set device
        device = self.device

        # Initialize hyperparameters from config
        model_config = self.config['model']
        seed = model_config['seed']
        epochs = model_config['epochs']
        batch_size = model_config['batch_size']
        experiment_name = model_config['experiment_name']
        experiment_description = model_config['experiment_description']
        log_progress = model_config['log_progress']
        train_fraction = model_config.get('train_fraction', None)
        train_count = model_config.get('train_study_count', None)

        # Split dataset and create dataloaders
        train_dataset, tune_dataset = dataset.train_test_split(train_fraction=train_fraction,
                                                               train_study_count=train_count,
                                                               seed=seed)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        tune_loader = DataLoader(tune_dataset, batch_size=1)

        self.model.to(device)

        self.loss_fn = self.loss_class()
        # If the loss function is weighted, weight losses based on the classes' prevalance

        if torch.nn.modules.loss._WeightedLoss in self.loss_class.__bases__:
            # TODO calculate class weights
            self.loss_fn = self.loss_class(weight=None)

        if log_progress:
            # Set up the neptune experiment
            utils.initialize_neptune(self.config)
            experiment = neptune.create_experiment(name=experiment_name,
                                                   description=experiment_description,
                                                   params=self.config
                                                   )

            experiment.set_property('model', str(type(self.model)))

            # Track the baseline (always predicting the most common class)
            label_counts = tune_dataset.map_labels_to_counts().values()

            tune_baseline = max(label_counts) / sum(label_counts)
            neptune.log_metric('tune_baseline', tune_baseline)

        best_tune_loss = None

        for epoch in tqdm(range(epochs)):
            train_loss = 0
            train_correct = 0
            self.model.train()

            for batch in train_loader:
                expression, labels = batch
                expression = expression.float().to(device)
                labels = labels.to(device)

                self.optimizer.zero_grad()
                output = self.model(expression)

                loss = self.loss_fn(output.unsqueeze(-1), labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += utils.count_correct(output, labels)
                # TODO f1 score

            with torch.no_grad():
                self.model.eval()

                tune_loss = 0
                tune_correct = 0

                for batch in tune_loader:
                    expression, labels = batch
                    expression = expression.float().to(device)
                    labels = labels.to(device)

                    output = self.model(expression)

                    tune_loss += self.loss_fn(output.unsqueeze(-1), labels).item()
                    tune_correct += utils.count_correct(output, labels)
                    # TODO f1 score

            train_acc = train_correct / len(train_dataset)
            tune_acc = tune_correct / len(tune_dataset)

            if log_progress:
                neptune.log_metric('train_loss', epoch, train_loss)
                neptune.log_metric('train_acc', epoch, train_acc)
                neptune.log_metric('tune_loss', epoch, tune_loss)
                neptune.log_metric('tune_acc', epoch, tune_acc)

            # Save model if applicable
            if 'save_path' in model_config:
                if best_tune_loss is None or tune_loss < best_tune_loss:
                    best_tune_loss = tune_loss
                    self.save_model(model_config['save_path'])

        return self

    def predict(self, dataset: UnlabeledDataset) -> np.ndarray:
        """
        Predict the labels for an unlabeled dataset

        Arguments
        ---------
        dataset: The unlabeled data whose labels should be predicted

        Returns
        -------
        predictions: A numpy array of predictions
        """
        data = dataset.get_all_data()
        X = torch.Tensor(data).float().to(self.device)

        self.model.eval()
        output = self.model(X)
        predictions = utils.sigmoid_to_predictions(output)
        return predictions

    def get_parameters(self) -> Iterable[torch.Tensor]:
        return copy.deepcopy(self.model.state_dict())

    def load_parameters(self, parameters: dict) -> "PytorchSupervised":
        self.model.load_state_dict(parameters)
        return self
