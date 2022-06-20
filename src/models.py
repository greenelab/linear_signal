""" A module containing the models to be trained on gene expression data """

import copy
import itertools
import pickle
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Union, Iterable, Tuple, Any

import neptune.new as neptune
import numpy as np
import sklearn.linear_model
import sklearn.decomposition
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import SelfAttention
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from datasets import LabeledDataset, UnlabeledDataset, MixedDataset, ExpressionDataset


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
    def fit(self, dataset: LabeledDataset, run: neptune.Run) -> "ExpressionModel":
        """
        Train a model using the given labeled data

        Arguments
        ---------
        dataset: The labeled data for use in training
        run: An object for logging training data if applicable

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
    def predict_proba(self, dataset: LabeledDataset) -> np.ndarray:
        """
        Return the raw predictions for each item in an unlabeled dataset

        Arguments
        ---------
        dataset: The data the model should be applied to

        Returns
        -------
        outputs: The raw model probabilities for the input
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
    """
    A logistic regression implementation designed to hew closely to the skl defaults
    """

    def __init__(self,
                 seed: int,
                 solver: str,
                 l2_penalty: float = None,
                 penalty_type: str = 'none',
                 **kwargs,
                 ) -> None:
        """
        The initializer for the LogisticRegression class

        Arguments
        ---------
        seed: The random seed to use in training
        l2_penalty: The inverse of the degree to which weights should be penalized
        solver: The method to use to optimize the loss
        """
        if penalty_type == 'none':
            self.model = sklearn.linear_model.LogisticRegression(random_state=seed,
                                                                 class_weight='balanced',
                                                                 penalty='none',
                                                                 solver=solver,
                                                                 multi_class='multinomial')
        else:
            self.model = sklearn.linear_model.LogisticRegression(random_state=seed,
                                                                 class_weight='balanced',
                                                                 penalty=penalty_type,
                                                                 C=l2_penalty,
                                                                 solver=solver,
                                                                 multi_class='multinomial')

    def fit(self, dataset: LabeledDataset, run: neptune.Run = None) -> "LogisticRegression":
        """
        Train a model using the given labeled data

        Arguments
        ---------
        dataset: The labeled data for use in training
        run: An object for logging training data if applicable

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

    def predict_proba(self, dataset: LabeledDataset) -> np.ndarray:
        """
        Return the raw predictions for each item in an unlabeled dataset

        Arguments
        ---------
        dataset: The data the model should be applied to

        Returns
        -------
        outputs: The raw model probabilities for the input
        """
        X, _ = dataset.get_all_data()
        return self.model.predict_proba(X)

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


class ThreeLayerWideBottleneck(nn.Module):
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
        super(ThreeLayerWideBottleneck, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 2)
        self.fc3 = nn.Linear(input_size // 2, output_size)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


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
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class PytorchLR(nn.Module):
    """ A pytorch implementation of logistic regression"""
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
        super(PytorchLR, self).__init__()

        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)

        return x


class FiveLayerImputation(nn.Module):
    """An imputation model based off the DeepClassifier (five-layer) model"""
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
        super(FiveLayerImputation, self).__init__()

        DROPOUT_PROB = .5

        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.bn1 = nn.BatchNorm1d(input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 2)
        self.bn2 = nn.BatchNorm1d(input_size // 2)
        self.fc3 = nn.Linear(input_size // 2, input_size // 2)
        self.bn3 = nn.BatchNorm1d(input_size // 2)
        self.fc4 = nn.Linear(input_size // 2, input_size // 4)
        self.bn4 = nn.BatchNorm1d(input_size // 4)
        self.fc5 = nn.Linear(input_size // 4, output_size)
        self.dropout = nn.Dropout(p=DROPOUT_PROB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout(x)

        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout(x)

        x = self.fc5(x)

        return x

    def get_final_layer(self):
        """ Return the last layer in the network for use by the PytorchImpute class """
        return self.fc5

    def set_final_layer(self, new_layer: nn.Module):
        """ Overwrite the final layer of the model with the layer passed in """
        self.fc5 = new_layer


class ThreeLayerImputation(nn.Module):
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
        super(ThreeLayerImputation, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size//2)
        self.fc2 = nn.Linear(input_size//2, input_size//2)
        self.fc3 = nn.Linear(input_size//2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_final_layer(self):
        """ Return the last layer in the network for use by the PytorchImpute class """
        return self.fc3

    def set_final_layer(self, new_layer: nn.Module):
        """ Overwrite the final layer of the model with the layer passed in """
        self.fc3 = new_layer


class PerformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        """
        dim: The embedding dimension of the data. For gene expression this will be 1
        num_heads: The number of heads to use in multi-headed attention
        """
        super(PerformerBlock, self).__init__()

        self.attn = SelfAttention(dim=dim,
                                  dim_head=64,
                                  heads=num_heads,
                                  causal=False)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out = self.attn(x)

        # Include a residual connection by adding the input and the attention output
        unnormalized_result = x + attn_out

        # TODO decide whether normalization helps
        # normalized_output = self.layer_norm(unnormalized_result)

        return unnormalized_result


class ImputePerformer(nn.Module):
    def __init__(self,
                 dim: int,
                 layer_count: int,
                 input_size: int,
                 num_heads: int = 8,
                 **kwargs):
        """
        Model initialization function

        Arguments
        ---------
        dim: The number of times to copy the data. Dim % heads must be 0
        layer_count: The number of blocks to include in the model
        input_size: The number of genes in the dataset. Acts like seq_len would if this were nlp
        num_heads: The number of heads to use in multi-headed attention
        """
        super(ImputePerformer, self).__init__()
        attn_block = PerformerBlock(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList(self.clone(attn_block, layer_count))
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)

    def clone(self, module: nn.Module, num_layers: int):
        "Produce N identical layers. From https://nlp.seas.harvard.edu/2018/04/03/attention.html"
        return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Duplicate the data along the "embedding dimension"
        x = x.unsqueeze(2).repeat(1, 1, self.dim)
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        attn_out = x
        # Add heads together to combine to keep memory from exploding
        flattened_attn = torch.sum(attn_out, 2, keepdim=False)

        x = F.relu(self.fc1(flattened_attn))
        x = self.fc2(x)
        return x

    def get_final_layer(self):
        """ Return the last layer in the network for use by the PytorchImpute class """
        return self.dec

    def set_final_layer(self, new_layer: nn.Module):
        """ Overwrite the final layer of the model with the layer passed in """
        self.dec = new_layer


class FCBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(FCBlock, self).__init__()
        DROPOUT_PROB = .5
        self.fc = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(p=DROPOUT_PROB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc(x))
        x = self.bn(x)
        x = self.dropout(x)
        return x


class GeneralClassifier(nn.Module):
    def __init__(self,
                 input_size: int,
                 intermediate_layers: int,
                 output_size: int,
                 **kwargs):
        """
        Model initialization function

        Arguments
        ---------
        input_size: The number of features in the dataset
        intermediate_layers: The number of layers in the middle of the model. The total number
                             of layers will be intermediate_layers + 2
        output_size: The number of classes to predict
        """
        super(GeneralClassifier, self).__init__()

        self.l1 = FCBlock(input_size, input_size // 2)

        intermediate_list = []
        for _ in range(intermediate_layers):
            intermediate_list.append(FCBlock(input_size // 2, input_size // 2))

        self.intermediate = nn.ModuleList(intermediate_list)

        self.output = nn.Linear(input_size // 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        for layer in self.intermediate:
            x = layer(x)
        x = self.output(x)

        return x


class DeepClassifier(nn.Module):
    """ A deep neural net for use in wrappers like PytorchSupervised"""
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
        super(DeepClassifier, self).__init__()

        DROPOUT_PROB = .5

        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.bn1 = nn.BatchNorm1d(input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 2)
        self.bn2 = nn.BatchNorm1d(input_size // 2)
        self.fc3 = nn.Linear(input_size // 2, input_size // 2)
        self.bn3 = nn.BatchNorm1d(input_size // 2)
        self.fc4 = nn.Linear(input_size // 2, input_size // 4)
        self.bn4 = nn.BatchNorm1d(input_size // 4)
        self.fc5 = nn.Linear(input_size // 4, output_size)
        self.dropout = nn.Dropout(p=DROPOUT_PROB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout(x)

        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout(x)

        x = self.fc5(x)

        return x


class PytorchImpute(ExpressionModel):
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
                 early_stopping_patience: int = 7,
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
        early_stopping_patience: The number of epochs to wait before stopping early
                                 if loss doesn't improve
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
        self.log_progress = log_progress
        self.train_fraction = train_fraction
        self.train_count = train_count
        self.loss_fn = self.loss_class(reduction='sum')
        self.save_path = save_path
        self.early_stopping_patience = early_stopping_patience
        self.model_name = model_name

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        model_class = get_model_by_name(model_name)
        self.model = model_class(**kwargs)

        self.optimizer = optimizer_class(self.model.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay)

        self.device = torch.device(device)

    def to_classifier(self,
                      num_classes: int,
                      classification_loss_name: str) -> "TransferClassifier":
        """
        Create a TransferClassifier object from the trained model by chopping off the final layer
        and replacing it with a classifier layer


        Arguments
        ---------
        num_classes: The number of classes the classifier should predict
        classification_loss_name: The name of the loss function to be used by the classifier

        Returns
        -------
        classifier: The resulting classifier model

        """
        if hasattr(self.model, 'get_final_layer'):
            final_layer = self.model.get_final_layer()
        else:
            sys.stderr.write('Warning: the model used in imputation does not have a ')
            sys.stderr.write('get_final_layer function\n')

            raise NotImplementedError

        intermediate_dimension = final_layer.in_features

        new_layer = nn.Linear(intermediate_dimension, num_classes)
        if hasattr(self.model, 'set_final_layer'):
            self.model.set_final_layer(new_layer)
        else:
            sys.stderr.write('Warning: the model used in imputation does not have a ')
            sys.stderr.write('set_final_layer function\n')

            raise NotImplementedError

        model_config = self.config
        model_config['pretrained_model'] = self.model

        # This line creates a dependency between this function and the TransferClassifier init
        # function that I don't really like. I can't think of a cleaner way to do it though
        model_config['loss_name'] = classification_loss_name
        if 'self' in model_config:
            del(model_config['self'])

        # Initialize the TransferClassifier
        new_model = TransferClassifier(**model_config)

        return new_model

    def free_memory(self) -> None:
        """
        The model subclass and optimizer used by PytorchImpute don't release their
        GPU memory by default when the main class is deleted. This function fixes that

        See https://github.com/greenelab/saged/issues/9 for more details
        """
        del self.model
        del self.optimizer

    @classmethod
    def load_model(cls,
                   checkpoint_path: str,
                   **kwargs
                   ) -> "PytorchImpute":
        """
        Read a pickled model from a file and return it

        Arguments
        ---------
        checkpoint_path: The location where the model is stored

        Returns
        -------
        model: The loaded model
        """
        model = cls(**kwargs)

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

    def mask_input_(self,
                    input_tensor: torch.Tensor,
                    fraction_masked: float = .1) -> torch.Tensor:
        """
        Apply a mask to the given input, setting a random subset of the input
        to zero

        Arguments
        ---------
        input_tensor: The tensor to be masked
        fraction_masked: The percent of the inputs to be set to zero

        Returns
        -------
        masked_expression: The input with the mask applied to it
        """
        # Create a mask with 10 percent ones to select locations to be
        # set to zero
        mask = utils.generate_mask(input_tensor.shape, fraction_zeros=(1 - fraction_masked))
        mask = mask.to(self.device)

        # Zero out masked items
        masked_expression = input_tensor.masked_fill(mask, 0)

        return masked_expression

    def fit(self, dataset: LabeledDataset, run: neptune.Run = None) -> "PytorchImpute":
        """
        Train a model using the given labeled data

        Arguments
        ---------
        dataset: The labeled data for use in training
        run: An object for logging training data if applicable

        Returns
        -------
        results: The metrics produced during the training process

        Raises
        ------
        AttributeError: If train_count and train_fraction are both None
        """
        # Set device
        device = self.device

        best_model_state = None
        best_optimizer_state = None

        seed = self.seed
        epochs = self.epochs
        batch_size = self.batch_size
        experiment_name = self.experiment_name
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
        # For very small training sets the tune dataset will be empty
        tune_is_empty = False
        if len(tune_dataset) == 0:
            sys.stderr.write('Warning: Tune dataset is empty')
            tune_is_empty = True

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        tune_loader = DataLoader(tune_dataset, batch_size=1)

        self.model.to(device)

        if log_progress:
            run['name'] = experiment_name
            run['params'] = self.config
            run['model'] = str(type(self.model))

        best_tune_loss = None
        epochs_since_best = 0

        for epoch in tqdm(range(epochs)):
            train_loss = 0
            self.model.train()

            for batch in train_loader:
                expression = batch
                expression = expression.float().to(device)

                masked_expression = self.mask_input_(expression)

                self.optimizer.zero_grad()
                output = self.model(masked_expression)

                loss = self.loss_fn(output, expression)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            with torch.no_grad():
                self.model.eval()

                tune_loss = 0

                if not tune_is_empty:
                    for batch in tune_loader:
                        expression = batch
                        expression = expression.float().to(device)

                        masked_expression = self.mask_input_(expression)

                        output = self.model(masked_expression)

                        tune_loss += self.loss_fn(output, expression).item()

            if log_progress:
                run['train/loss'].log(train_loss / len(train_dataset))
                run['tune/loss'].log(tune_loss / len(tune_dataset))

            if not tune_is_empty:
                if best_tune_loss is None or tune_loss < best_tune_loss:
                    best_tune_loss = tune_loss
                    # Keep track of model state for the best model
                    best_model_state = {k: v.to('cpu') for k, v in self.model.state_dict().items()}
                    best_model_state = OrderedDict(best_model_state)
                    best_optimizer_state = {}

                    # Ideally we would use deepcopy here, but we need to get the tensors off the GPU
                    for key, value in self.optimizer.state_dict().items():
                        if isinstance(value, torch.Tensor):
                            best_optimizer_state[key] = value.to('cpu')
                        else:
                            best_optimizer_state[key] = value
                    best_optimizer_state = OrderedDict(best_optimizer_state)
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1

                if epochs_since_best >= self.early_stopping_patience:
                    break

        # Load model from state dict
        save_path = getattr(self, 'save_path', None)
        if not tune_is_empty:
            self.load_parameters(best_model_state)
            self.optimizer.load_state_dict(best_optimizer_state)
            if save_path is not None:
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

    def predict_proba(self, dataset: LabeledDataset) -> np.ndarray:
        """
        Return the raw predictions for each item in an unlabeled dataset

        Arguments
        ---------
        dataset: The data the model should be applied to

        Returns
        -------
        outputs: The raw model probabilities for the input
        """
        data, _ = dataset.get_all_data()
        X = torch.Tensor(data).float().to(self.device)

        self.model.eval()
        output = self.model(X)
        return output

    def evaluate(self, dataset: MixedDataset) -> float:
        """
        Return the loss for the validation dataset

        Arguments
        ---------
        dataset: The labeled dataset for use in evaluating the model

        Returns
        -------
        loss: The loss (usually mean squared error) of the dataset
        """
        data_loader = DataLoader(dataset, batch_size=1)
        with torch.no_grad():
            self.model.eval()

            total_loss = 0
            for batch in data_loader:
                expression = batch
                expression = expression.float().to(self.device)
                masked_expression = self.mask_input_(expression)
                output = self.model(masked_expression)

                total_loss += self.loss_fn(output, expression).item()

        return total_loss

    def get_parameters(self) -> Iterable[torch.Tensor]:
        return copy.deepcopy(self.model.state_dict())

    def load_parameters(self, parameters: dict) -> "PytorchSupervised":
        self.model.load_state_dict(parameters)
        return self


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
                 l2_penalty: float,
                 device: str,
                 seed: int,
                 epochs: int,
                 batch_size: int,
                 log_progress: bool,
                 experiment_name: str = None,
                 save_path: str = None,
                 train_fraction: float = None,
                 train_count: float = None,
                 clip_grads: bool = False,
                 early_stopping_patience: int = 7,
                 pretrained_model: nn.Module = None,
                 final_layer_name: str = None,
                 loss_weights: torch.Tensor = None,
                 full_batch_training: bool = False,
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
        l2_penalty: The weight decay for the optimizer
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
        clip_grads: A flag reflecting whether to perform clip gradients during training
        early_stopping_patience: The number of epochs to wait before stopping early
                                 if loss doesn't improve
        pretrained_model: If a model is passed here, it will be used instead of initializing
                          a new model
        final_layer_name: The name of the attribute in the model that stores the final layer
        loss_weights: Per-class weights for handling class imbalance
        full_batch_training: If this flag is True, ignore batch_size and train on full dataset in
                             each epoch
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
        self.log_progress = log_progress
        self.train_fraction = train_fraction
        self.train_count = train_count
        self.clip_grads = clip_grads
        self.save_path = save_path
        self.early_stopping_patience = early_stopping_patience
        self.final_layer_name = final_layer_name
        self.loss_weights = loss_weights
        self.full_batch_training = full_batch_training

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        if pretrained_model is None:
            model_class = get_model_by_name(model_name)
            self.model = model_class(**kwargs)
        else:
            self.model = pretrained_model

        if optimizer_class == torch.optim.LBFGS:
            self.optimizer = optimizer_class(self.model.parameters(),
                                             max_iter=100)
        else:
            self.optimizer = optimizer_class(self.model.parameters(),
                                             lr=lr,
                                             weight_decay=l2_penalty)

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

    def fit(self, dataset: LabeledDataset, run: neptune.Run = None) -> "PytorchSupervised":
        """
        Train a model using the given labeled data

        Arguments
        ---------
        dataset: The labeled data for use in training
        run: An object for logging training data if applicable

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
        # For very small training sets the tune dataset will be empty
        # TODO figure out how to more elegantly handle this when implementing early stopping
        tune_is_empty = False
        if len(tune_dataset) == 0:
            sys.stderr.write('Warning: Tune dataset is empty')
            tune_is_empty = True

        if self.full_batch_training:
            batch_size = len(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        tune_loader = DataLoader(tune_dataset, batch_size=1)

        self.model.to(device)

        self.loss_fn = self.loss_class()
        # If the loss function is weighted, weight losses based on the classes' prevalance
        if (torch.nn.modules.loss._WeightedLoss in self.loss_class.__bases__ and
           self.loss_weights is not None):
            self.loss_weights = self.loss_weights.to(device)
            self.loss_fn = self.loss_class(weight=self.loss_weights)

        if log_progress:
            run['experiment_name'] = self.experiment_name
            run['params'] = self.config
            run['model'] = str(type(self.model))

            # Track the baseline (always predicting the most common class)
            if not tune_is_empty:
                label_counts = tune_dataset.map_labels_to_counts().values()

                tune_baseline = max(label_counts) / sum(label_counts)
                run['tune_baseline', tune_baseline]

        best_tune_loss = None
        epochs_since_best = 0

        for epoch in tqdm(range(epochs)):
            train_loss = 0
            train_correct = 0
            self.model.train()

            for batch in train_loader:
                expression, labels = batch

                # Ignore singleton batches
                if len(expression) <= 1:
                    continue

                expression = expression.float().to(device)
                labels = labels.squeeze()

                # Single element batches get squeezed from 2d into 0d, so unsqueeze them a bit
                if labels.dim() == 0:
                    labels = labels.unsqueeze(-1)

                labels = labels.to(device)

                if type(self.optimizer) == torch.optim.LBFGS:
                    def closure():
                        # https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
                        self.optimizer.zero_grad()
                        output = self.model(expression)

                        loss = self.loss_fn(output, labels)

                        loss.backward()
                        return loss

                    loss = closure()
                    self.optimizer.step(closure)
                    output = self.model(expression)
                else:
                    self.optimizer.zero_grad()
                    output = self.model(expression)

                    loss = self.loss_fn(output, labels)

                    loss.backward()

                    if getattr(self, 'clip_grads', False):
                        nn.utils.clip_grad_norm_(self.model.parameters(), .01)

                    self.optimizer.step()

                train_loss += loss.item()
                train_correct += utils.count_correct(output, labels)

            with torch.no_grad():

                self.model.eval()

                tune_loss = 0
                tune_correct = 0

                if not tune_is_empty:
                    for batch in tune_loader:
                        expression, labels = batch
                        expression = expression.float().to(device)
                        labels = labels.to(device)

                        output = self.model(expression)

                        tune_loss += self.loss_fn(output.unsqueeze(-1), labels).item()
                        tune_correct += utils.count_correct(output, labels)

            train_acc = train_correct / len(train_dataset)
            if not tune_is_empty:
                tune_acc = tune_correct / len(tune_dataset)

            if log_progress:
                run['train/loss'].log(train_loss)
                run['train/acc'].log(train_acc)
                if not tune_is_empty:
                    run['tune/loss'].log(tune_loss)
                    run['tune/acc'].log(tune_acc)

            if not tune_is_empty:
                if best_tune_loss is None or tune_loss < best_tune_loss:
                    best_tune_loss = tune_loss
                    # Keep track of model state for the best model
                    best_model_state = {k: v.to('cpu') for k, v in self.model.state_dict().items()}
                    best_model_state = OrderedDict(best_model_state)
                    best_optimizer_state = {}

                    # Ideally we would use deepcopy here, but we need to get the tensors off the GPU
                    for key, value in self.optimizer.state_dict().items():
                        if isinstance(value, torch.Tensor):
                            best_optimizer_state[key] = value.to('cpu')
                        else:
                            best_optimizer_state[key] = value
                    best_optimizer_state = OrderedDict(best_optimizer_state)
                else:
                    epochs_since_best += 1

                if epochs_since_best >= self.early_stopping_patience:
                    break

        # Load model from state dict
        save_path = getattr(self, 'save_path', None)
        if not tune_is_empty:
            self.load_parameters(best_model_state)
            self.optimizer.load_state_dict(best_optimizer_state)
            if save_path is not None:
                self.save_model(save_path)

        else:
            print('Using model from final epoch; early stopping is off')

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

    def predict_proba(self, dataset: LabeledDataset) -> np.ndarray:
        """
        Return the raw predictions for each item in an unlabeled dataset

        Arguments
        ---------
        dataset: The data the model should be applied to

        Returns
        -------
        outputs: The raw model probabilities for the input
        """
        data, _ = dataset.get_all_data()
        X = torch.Tensor(data).float().to(self.device)

        self.model.eval()
        output = self.model(X)
        return output

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


class UnsupervisedModel(ABC):
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


class PseudolabelModel(PytorchSupervised):
    """
    A wrapper class implementing the ExpressionModel API while remaining modular enough
    to accept any supervised classifier implementing the nn.Module API.
    Extends the training logic in PytorchSupervised to use pseudolabeling on unlabeled data
    """
    def __init__(self, max_alpha, **kwargs):
        """
        Initialize the pseudolabeled model by calling the PytorchSupervised init function
        then adding on the alpha_max member variable
        """
        super().__init__(**kwargs)

        self.max_alpha = max_alpha

    def fit(self, dataset: MixedDataset, run: neptune.Run = None) -> "PseudolabelModel":
        """
        Train a model using the given labeled data

        Arguments
        ---------
        dataset: The labeled data for use in training
        run: An object for logging training data if applicable

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
        log_progress = self.log_progress

        train_count = None
        train_fraction = None
        train_fraction = getattr(self, 'train_fraction', None)
        if train_fraction is None:
            train_count = self.train_count

        labeled_data = dataset.get_labeled()
        unlabeled_data = dataset.get_unlabeled()

        # Split dataset and create dataloaders
        train_dataset, tune_dataset = labeled_data.train_test_split(train_fraction=train_fraction,
                                                                    train_study_count=train_count,
                                                                    seed=seed)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        tune_loader = DataLoader(tune_dataset, batch_size=1)
        unlabeled_loader = DataLoader(unlabeled_data, batch_size, shuffle=True, drop_last=True)

        self.model.to(device)

        self.loss_fn = self.loss_class()
        # If the loss function is weighted, weight losses based on the classes' prevalance

        if torch.nn.modules.loss._WeightedLoss in self.loss_class.__bases__:
            # TODO calculate class weights
            self.loss_fn = self.loss_class(weight=None)

        if log_progress:
            run['experiment_name'] = self.experiment_name
            run['params'] = self.config
            run['model'] = str(type(self.model))

            # Track the baseline (always predicting the most common class)
            label_counts = tune_dataset.map_labels_to_counts().values()

            tune_baseline = max(label_counts) / sum(label_counts)
            run['tune_baseline'] = tune_baseline

        best_tune_loss = None

        for epoch in tqdm(range(epochs)):

            # Alpha (the ratio of pseudolabel loss to label loss to use)
            # increases linearly across epochs. This allows the model to ignore pseudolabeling
            # information while the model is bad, and increases pseudolabels' impact when
            # the model has been trained on the input data for awhile
            if epochs > 1:
                progress = epoch / (epochs-1)
            else:
                progress = 0
            alpha = progress * self.max_alpha

            train_loss = 0
            train_correct = 0
            self.model.train()

            # Create an iterator that returns a labeled batch and an unlabeled batch
            if len(labeled_data) <= len(unlabeled_data):
                # If there are more unlabeled samples than labeled samples, just stop iterating
                # once you've seen each labeled sample once (zip stops at the end of the
                # shorter iterator)
                train_iterator = zip(train_loader, unlabeled_loader)
            else:
                # If there is more labeled data than unlabeled data, then loop the unlabeled
                # data iterator
                # NOTE: itertools.cycle eats a lot of memory so if you have a lot more labeled
                # data than unlabeled data this might start swapping
                train_iterator = zip(train_loader, itertools.cycle(unlabeled_data))

            for train_batch, unlabeled_expression in train_iterator:
                train_expression, train_labels = train_batch
                train_expression = train_expression.float().to(device)
                unlabeled_expression = unlabeled_expression.float().to(device)
                train_labels = train_labels.squeeze()

                # Single element batches get squeezed from 2d into 0d, so unsqueeze them a bit
                if train_labels.dim() == 0:
                    train_labels = train_labels.unsqueeze(-1)

                train_labels = train_labels.to(device)

                self.optimizer.zero_grad()
                train_output = self.model(train_expression)

                labeled_loss = self.loss_fn(train_output, train_labels)

                # Pseudolabel points and calculate their loss
                unlabeled_output = self.model(unlabeled_expression)
                unlabeled_preds = torch.argmax(unlabeled_output, axis=-1)
                unlabeled_loss = self.loss_fn(unlabeled_output, unlabeled_preds)

                loss = labeled_loss + alpha * unlabeled_loss

                loss.backward()
                self.optimizer.step()

                train_loss += labeled_loss.item()
                train_correct += utils.count_correct(train_output, train_labels)

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

            train_acc = train_correct / len(train_dataset)
            tune_acc = tune_correct / len(tune_dataset)

            if log_progress:
                run['train/loss'].log(train_loss)
                run['train/acc'].log(train_acc)
                run['tune/loss'].log(tune_loss)
                run['tune/acc'].log(tune_acc)

            # Save model if applicable
            save_path = getattr(self, 'save_path', None)
            if save_path is not None:
                if best_tune_loss is None or tune_loss < best_tune_loss:
                    best_tune_loss = tune_loss
                    self.save_model(save_path)

        return self


class TransferClassifier(PytorchSupervised):
    def __init__(self,
                 pretrained_model: nn.Module,
                 **kwargs,):
        """
        Standard model init function for a supervised model

        Arguments
        ---------
        pretrained_model: A model trained via imputing gene expression data
        optimizer_name: The name of the optimizer to be used to train the classifier
        **kwargs: Arguments for use in the underlying classifier

        Notes
        -----
        Either `train_count` or `train_fraction` should be None but not both
        """
        super(TransferClassifier, self).__init__(pretrained_model=pretrained_model, **kwargs)
