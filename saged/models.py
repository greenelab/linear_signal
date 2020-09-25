""" A module containing the models to be trained on gene expression data """

import copy
import pickle
from abc import ABC, abstractmethod
from typing import Union, Iterable, Tuple, Any

import neptune
import numpy as np
import sklearn.linear_model
import sklearn.decomposition
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import saged.utils as utils
from saged.datasets import LabeledDataset, UnlabeledDataset, MixedDataset, ExpressionDataset


def get_model_by_name(model_name: str) -> Any:
    """
    This function invokes old magic to get a model class from the current file dynamically.
    In python the answer to 'How do I get a class from the current file dynamically' is
    'Dump all the global variables for the file, it will be there somewhere'
    https://stackoverflow.com/questions/734970/python-reference-to-a-class-from-a-string

    Arguments
    ---------
    model_name: The name of the class object to return

    Returns
    -------
    model_class: The class the model specified e.g. PCA or LogisticRegression
    """
    model_class = globals()[model_name]

    return model_class


def embed_data(unsupervised_config: dict,
               all_data: MixedDataset,
               train_data: LabeledDataset,
               unlabeled_data: UnlabeledDataset,
               val_data: LabeledDataset,
               ) -> Tuple[LabeledDataset, LabeledDataset, "UnsupervisedModel"]:
    """
    Initialize an unsupervised model and use it to reduce the dimensionality of the data

    Arguments
    ---------
    unsupervised_config: The path to the yml file detailing how to initialize the
                         unsupervised model
    all_data: The object storing the data for the entire dataset
    train_data: The subset of the data to be used for training
    unlabeled_data: The subset of the data that doesn't have labels
    val_data: The subset of the data that will be used for validation. To avoid data leakage, the
              validation data will not be used to train the unsupervised embedding, but will be
              embedded
    Returns
    -------
    train_data: The embedded training data
    val_data: The embedded validation data
    unsupervised_model: The fitted version of the model
    """
    # Initialize the unsupervised model
    unsupervised_model_type = unsupervised_config.pop('name')
    UnsupervisedClass = get_model_by_name(unsupervised_model_type)

    unsupervised_model = UnsupervisedClass(**unsupervised_config)

    # Get all data not held in the val split
    available_data = all_data.subset_to_samples(train_data.get_samples() +
                                                unlabeled_data.get_samples())

    # Embed the training data
    unsupervised_model.fit(available_data)
    train_data = unsupervised_model.transform(train_data)

    # Embed the validation data
    val_data = unsupervised_model.transform(val_data)

    # Reset filters on all_data which were changed to create available_data
    all_data.reset_filters()

    return train_data, val_data, unsupervised_model


class ExpressionModel(ABC):
    """
    A model API similar to the scikit-learn API that will specify the
    base acceptable functions for models in this module's benchmarking code
    """

    def __init__(self,
                 config: dict) -> None:
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

    def free_memory(self) -> None:
        """
        Some models need help freeing the memory allocated to them. Others do not.
        This function is a placeholder that can be overridden by inheriting classes
        if needed. PytorchSupervised is a good example of what a custom free_memory function
        does.
        """
        pass

    @abstractmethod
    def fit(self, dataset: LabeledDataset) -> "ExpressionModel":
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
        Predict the labels for a dataset

        Arguments
        ---------
        dataset: The unlabeled data whose labels should be predicted

        Returns
        -------
        predictions: A numpy array of predictions
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, dataset: LabeledDataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the predicted and true labels for a dataset

        Arguments
        ---------
        dataset: The labeled dataset for use in evaluating the model

        Returns
        -------
        predictions: A numpy array of predictions
        labels: The true labels to compare the predictions against
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

    def __init__(self,
                 seed: int,
                 **kwargs,
                 ) -> None:
        """
        The initializer for the LogisticRegression class

        Arguments
        ---------
        seed: The random seed to use in training
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

    def evaluate(self, dataset: LabeledDataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the predicted and true labels for a dataset

        Arguments
        ---------
        dataset: The labeled dataset for use in evaluating the model

        Returns
        -------
        predictions: A numpy array of predictions
        labels: The true labels to compare the predictions against
        """
        X, y = dataset.get_all_data()
        return self.model.predict(X), y

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
    def load_model(classobject, model_path: str, **kwargs):
        """
        Read a pickeled model from a file and return it

        Arguments
        ---------
        model_path: The location where the model is stored
        **kwargs: To be consistent with the API this function takes in config info even though
                  it doesn't need it

        Returns
        -------
        model: The model saved at `model_path`
        """
        with open(model_path, 'rb') as model_file:
            return pickle.load(model_file)


class ThreeLayerClassifier(nn.Module):
    """ A basic three layer neural net for use in wrappers like PytorchSupervised"""
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 **kwargs):
        """
        Model initialization function

        Arguments
        ---------
        input_size: The number of features in the dataset
        output_size: The number of classes to predict
        """
        super(ThreeLayerClassifier, self).__init__()

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
                 optimizer_name: str,
                 loss_name: str,
                 model_name: str,
                 lr: float,
                 weight_decay: float,
                 device: str,
                 seed: int,
                 epochs: int,
                 batch_size: int,
                 log_progress: bool,
                 experiment_name: str = None,
                 experiment_description: str = None,
                 save_path: str = None,
                 train_fraction: float = None,
                 train_count: float = None,
                 **kwargs,
                 ) -> None:
        """
        Standard model init function for a supervised model

        Arguments
        ---------
        optimizer_name: The name of the optimizer class to be used when training the model
        loss_name: The loss function class to use
        model_name: The type of classifier to use
        lr: The learning rate for the optimizer
        weight_decay: The weight decay for the optimizer
        device: The name of the device to train on (typically 'cpu', 'cuda', or 'tpu')
        seed: The random seed to use in stochastic operations
        epochs: The number of epochs to train the model
        batch_size: The number of items in each training batch
        log_progress: True if you want to use neptune to log progress, otherwise False
        experiment_name: A short name for the experiment you're running for use in neptune logs
        experiment_description: A description for the experiment you're running
        save_path: The path to save the model to
        train_fraction: The percent of samples to use in training
        train_count: The number of studies to use in training
        **kwargs: Arguments for use in the underlying classifier

        Notes
        -----
        Either `train_count` or `train_fraction` should be None but not both
        """
        # A piece of obscure python, this gets a dict of all python local variables.
        # Since it is called at the start of a function it gets all the arguments for the
        # function as if they were passed in a dict. This is useful, because we can feed
        # self.config to neptune to keep track of all our run's parameters
        self.config = locals()

        optimizer_class = getattr(torch.optim, optimizer_name)
        self.loss_class = getattr(nn, loss_name)

        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.log_progress = log_progress
        self.train_fraction = train_fraction
        self.train_count = train_count

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        # We're invoking the old magic now. In python the answer to 'How do I get a class from
        # the current file dynamically' is 'Dump all the global variables for the file, it will
        # be there somewhere'
        # https://stackoverflow.com/questions/734970/python-reference-to-a-class-from-a-string
        model_class = get_model_by_name(model_name)
        self.model = model_class(**kwargs)

        self.optimizer = optimizer_class(self.model.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay)

        self.device = torch.device(device)

    def free_memory(self) -> None:
        """
        The model subclass and optimizer used by PytorchSupervised don't release their
        GPU memory by default when the main class is deleted. This function fixes that

        See https://github.com/greenelab/saged/issues/9 for more details
        """
        del self.model
        del self.optimizer

    @classmethod
    def load_model(classobject,
                   checkpoint_path: str,
                   **kwargs
                   ) -> "PytorchSupervised":
        """
        Read a pickled model from a file and return it

        Arguments
        ---------
        checkpoint_path: The location where the model is stored

        Returns
        -------
        model: The loaded model
        """
        model = classobject(**kwargs)

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

        Raises
        ------
        AttributeError: If train_count and train_fraction are both None
        """
        # Set device
        device = self.device

        seed = self.seed
        epochs = self.epochs
        batch_size = self.batch_size
        experiment_name = self.experiment_name
        experiment_description = self.experiment_description
        log_progress = self.log_progress

        train_count = None
        train_fraction = None
        train_fraction = getattr(self, 'train_fraction', None)
        if train_fraction is None:
            train_count = self.train_count

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
                labels = labels.squeeze()

                # Single element batches get squeezed from 2d into 0d, so unsqueeze them a bit
                if labels.dim() == 0:
                    labels = labels.unsqueeze(-1)

                labels = labels.to(device)

                self.optimizer.zero_grad()
                output = self.model(expression)

                loss = self.loss_fn(output, labels)
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
            save_path = getattr(self, 'save_path', None)
            if save_path is not None:
                if best_tune_loss is None or tune_loss < best_tune_loss:
                    best_tune_loss = tune_loss
                    self.save_model(save_path)

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

    def evaluate(self, dataset: LabeledDataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the predicted and true labels for a dataset

        Arguments
        ---------
        dataset: The labeled dataset for use in evaluating the model

        Returns
        -------
        predictions: A numpy array of predictions
        labels: The true labels to compare the predictions against
        """
        X, y = dataset.get_all_data()
        X = torch.Tensor(X).float().to(self.device)

        self.model.eval()
        output = self.model(X)
        predictions = utils.sigmoid_to_predictions(output)
        predictions = predictions.cpu().numpy()
        return predictions, y

    def get_parameters(self) -> Iterable[torch.Tensor]:
        return copy.deepcopy(self.model.state_dict())

    def load_parameters(self, parameters: dict) -> "PytorchSupervised":
        self.model.load_state_dict(parameters)
        return self


class UnsupervisedModel():
    """
    A model API defining the behavior of unsupervised models. Largely follows the sklearn model api
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
    def fit(self, dataset: Union[UnlabeledDataset, MixedDataset]) -> "UnsupervisedModel":
        """
        Train a model using the given unlabeled data

        Arguments
        ---------
        dataset: The labeled data for use in training

        Returns
        -------
        self: The trained version of the model
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataset: UnlabeledDataset) -> UnlabeledDataset:
        """
        Use the learned embedding from the model to embed the given dataset

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

    def fit_transform(self, dataset: UnlabeledDataset) -> UnlabeledDataset:
        """
        Learn an embedding from the given data, then return the embedded data

        Arguments
        ---------
        dataset: The unlabeled data whose embedding should be learned

        Returns
        -------
        embedded_data: The dataset returned by the transform function
        """
        self.fit(dataset)
        return self.transform(dataset)


class PCA(UnsupervisedModel):
    """
    A wrapper for the sklearn PCA function
    """
    def __init__(self,
                 n_components: int,
                 seed: int = 42,
                 **kwargs) -> None:
        """
        PCA initialization function

        Arguments
        ---------
        n_components: The number of principal components to keep. That is to say, the dimenstion
                      to which the input will be embedded to
        seed: The random seed
        """
        self.model = sklearn.decomposition.PCA(n_components=n_components,
                                               random_state=seed)

    @classmethod
    def load_model(classobject, model_path: str, **kwargs):
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

    def fit(self, dataset: Union[UnlabeledDataset, MixedDataset]) -> "UnsupervisedModel":
        """
        Train a model using the given unlabeled data

        Arguments
        ---------
        dataset: The labeled data for use in training

        Returns
        -------
        self: The trained version of the model
        """
        if issubclass(type(dataset), LabeledDataset):
            X = dataset.get_all_data()[0]
        else:
            X = dataset.get_all_data()
        self.model = self.model.fit(X)

        return self

    def transform(self, dataset: ExpressionDataset) -> ExpressionDataset:
        """
        Use the learned embedding from the model to embed the given dataset

        Arguments
        ---------
        dataset: The unlabeled data whose labels should be predicted

        Returns
        -------
        dataset_copy: The transformed version of a copy of the dataset passed in
        """
        if issubclass(type(dataset), LabeledDataset):
            X = dataset.get_all_data()[0]
        else:
            X = dataset.get_all_data()
        X_embedded = self.model.transform(X)

        # This is necessary to match the sklearn API by not overwriting the original dataset
        # There may be an edge case where this breaks the get_studies function since the
        # transformed and original dataset will share references to `is_changed`, but I don't
        # think that will be the case.
        dataset_copy = copy.copy(dataset)

        dataset_copy.set_all_data(X_embedded.T)

        return dataset_copy

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
        with open(out_path, 'wb') as out_file:
            pickle.dump(self, out_file)
