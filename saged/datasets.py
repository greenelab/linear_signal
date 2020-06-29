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
        pass

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
        idx: The index of the given item

        Returns
        -------
        X: The gene expression data for the given index in a genes x 1 array
        y: The label corresponding to the sample in X
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_data(self) -> Tuple[np.array, np.array]:
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

    def subset_studies(self, fraction: float = None, num_studies: int = None) -> float:
        """
        This method is similar to `subset_samples`, but removes entire studies until the
        fraction of data remaining is less than the fraction passed in.

        As this will almost never result in the fraction being met exactly, the data fraction
        that actually comes out of the subsetting will be returned.

        Either `fraction` or `num_studies` must be specified. If both are specified,
        `num_studies` will be given preference.

        Arguments
        ---------
        fraction: The fraction of the samples to keep
        num_studies: The number of studies to keep

        Returns
        -------
        true_fraction: The fraction of the samples that were actually kept
        """
        raise NotImplementedError


    @abstractmethod
    # For more info on using a forward reference for the type, see
    # https://github.com/python/mypy/issues/3661#issuecomment-313157498
    def get_cv_splits(self, num_splits: int) -> Sequence["ExpressionDataset"]:
        """
        Split the dataset into a list of smaller dataset objects with a roughly equal
        number of samples in each.

        If multiple studies are present in the dataset, each study should only
        be present in a single fold

        Arguments
        ---------
        num_splits: The number of groups to split the dataset into

        Returns
        -------
        subsets: A list of datasets, each composed of fractions of the original
        """
        raise NotImplementedError

    @abstractmethod
    def train_test_split(self,
                         train_fraction: float = None,
                         train_study_count: int = None
                        ) -> Sequence["ExpressionDataset"]:
        """
        Split the dataset into two portions, as seen in scikit-learn's `train_test_split`
        function.

        If multiple studies are present in the dataset, each study should only
        be present in one of the two portions.

        Either `train_fraction` or `train_study_count` must be specified. If both
        are specified, then `train_study_count` takes precedence.

        Arguments
        ---------
        train_fraction: The minimum fraction of the data to be used as training data.
            In reality, the fraction won't be met entirely due to the constraint of
            preserving studies.
        train_study_count: The number of studies to be included in the training set

        Returns
        -------
        train: The dataset with around the amount of data specified by train_fraction
        test: The dataset with the remaining data
        """
        raise NotImplementedError


class LabeledDataset(ExpressionDataset):
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


class UnlabeledDataset(ExpressionDataset):
    @abstractmethod
    def __getitem__(self, idx: int) -> np.array:
        """
        Allows access into the dataset to select a single datapoint

        Arguments
        ---------
        idx: The index of the given item

        Returns
        -------
        X: The gene expression data for the given index in a genes x 1 array
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_data(self) -> np.array:
        """
        Returns all the expression data from the dataset in the
        form of a numpy array

        Returns
        -------
        X: The gene expression data in a genes x samples array
        """
        raise NotImplementedError
