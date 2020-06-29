""" This file contains dataset objects for manipulating gene expression data """
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List

import numpy as np


class ExpressionDataset(ABC):
    """
    The base dataset defining the API for datasets in this project. This class is
    based on the Dataset object from pytorch
    (https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
    """
    @abstractmethod
    def __init__(self) -> None:
        """
        Abstract initializer. When implemented this function should take in a datasource
        and use it to initialize the class member variables
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the length of the dataset for use in determining
        the size of an epoch

        Arguments
        ---------

        Returns
        -------
        length: The number of samples currently available in the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """
        Allows access into the dataset to select a single datapoint and its associated label

        Arguments
        ---------

        Returns
        -------
        Returns
        -------
        X: The gene expression data for the given index in a genes x 1 array
        y: The label corresponding to the sample in X
        """
        raise NotImplementedError

    @abstractmethod
    def get_sklearn_data(self) -> Tuple[np.array, np.array]:
        """
        Returns all the expression data and labels from the dataset in the
        form of an (X,y) tuple where both X and y are numpy arrays

        Returns
        -------
        X: The gene expression data in a genes x samples array
        y: The label corresponding to each sample in X
        """
        raise NotImplementedError

    @abstractmethod
    def subset_samples(self, fraction: float) -> None:
        """
        Limit the amount of data available to be returned in proportion to
        the total amount of data that was present in the dataset.
        That is to say that calling subset_data_by_fraction twice will not cause the Dataset
        object to have multiplicatively less data, it will have the fraction of data specified
        in the second call

        Arguments
        ---------
        fraction: The fraction of the samples to keep
        """
        raise NotImplementedError

    def subset_studies(self, fraction: float) -> float:
        """
        This method is similar to `subset_samples`, but removes entire studies until the
        fraction of data remaining is less than the fraction passed in.

        As this will almost never result in the fraction being met exactly, the data fraction
        that actually comes out of the subsetting will be returned

        Arguments
        ---------
        fraction: The fraction of the samples to keep

        Returns
        -------
        true_fraction: The fraction of the samples that were actually kept
        """
        raise NotImplementedError

    @abstractmethod
    def subset_samples_for_label(self, fraction: float, label: str) -> None:
        """
        Limit the number of samples available for a single label.
        For example, if you wanted to use only ten percent of the sepsis expression
        samples across all studies, you would use this function.

        Arguments
        ---------
        fraction: The fraction of the samples to keep
        label: The category of data to apply this subset to
        """
        raise NotImplementedError

    @abstractmethod
    def subset_samples_to_labels(self, labels: List[str]) -> None:
        """
        Keep only the samples corresponding to the labels passed in.
        Stacks with other labels; if `subset_samples_for_label`

        Arguments
        ---------
        labels: The label or labels of samples to keep
        """
        raise NotImplementedError

    @abstractmethod
    # For more info on using a forward reference for the type, see
    # https://github.com/python/mypy/issues/3661#issuecomment-313157498
    def get_cv_splits(self, num_splits) -> Sequence["ExpressionDataset"]:
        """
        Split the dataset into a list of smaller dataset objects with a roughly equal
        number of samples in each.

        If multiple studies are present in the dataset, each dataset should only
        have data from a single study

        Arguments
        ---------
        num_splits: The number of groups to split the dataset into

        Returns
        -------
        subsets: A list of datasets, each composed of fractions of the original
        """
        raise NotImplementedError
