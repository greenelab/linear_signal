""" This file contains dataset objects for manipulating gene expression data """
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Tuple, List, Union, Set

import numpy as np

import utils


class ExpressionDataset(ABC):
    """
    The base dataset defining the API for datasets in this project. This class is
    based on the Dataset object from pytorch
    (https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset)
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
    def subset_samples(self, fraction: float, seed: int = 42) -> "ExpressionDataset":
        """
        Limit the amount of data available to be returned in proportion to
        the total amount of data that was present in the dataset.
        That is to say that calling subset_data_by_fraction twice will not cause the Dataset
        object to have multiplicatively less data, it will have the fraction of data specified
        in the second call

        Arguments
        ---------
        fraction: The fraction of the samples to keep
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        self: The object after subsetting
        """
        raise NotImplementedError

    def subset_studies(self,
                       fraction: float = None,
                       num_studies: int = None,
                       seed: int = 42) -> "ExpressionDataset":
        """
        This method is similar to `subset_samples`, but removes entire studies until the
        fraction of data remaining is less than the fraction passed in.

        Either `fraction` or `num_studies` must be specified. If both are specified,
        `num_studies` will be given preference.

        Arguments
        ---------
        fraction: The fraction of the samples to keep
        num_studies: The number of studies to keep
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        self: The object after subsetting

        Raises
        ------
        ValueError: Neither fraction nor num_studies were specified
        """
        raise NotImplementedError

    @abstractmethod
    # For more info on using a forward reference for the type, see
    # https://github.com/python/mypy/issues/3661#issuecomment-313157498
    def get_cv_splits(self, num_splits: int, seed: int = 42) -> Sequence["ExpressionDataset"]:
        """
        Split the dataset into a list of smaller dataset objects with a roughly equal
        number of samples in each.

        If multiple studies are present in the dataset, each study should only
        be present in a single fold

        Arguments
        ---------
        num_splits: The number of groups to split the dataset into
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        subsets: A list of datasets, each composed of fractions of the original
        """
        raise NotImplementedError

    @abstractmethod
    def train_test_split(self,
                         train_fraction: float = None,
                         train_study_count: int = None,
                         seed: int = 42,
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
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        train: The dataset with around the amount of data specified by train_fraction
        test: The dataset with the remaining data
        """
        raise NotImplementedError


class LabeledDataset(ExpressionDataset):
    @abstractmethod
    def subset_samples_for_label(self, fraction: float,
                                 label: str,
                                 seed: int = 42,
                                 ) -> "LabeledDataset":
        """
        Limit the number of samples available for a single label.
        For example, if you wanted to use only ten percent of the sepsis expression
        samples across all studies, you would use this function.

        Arguments
        ---------
        fraction: The fraction of the samples to keep
        label: The category of data to apply this subset to
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        self: The subsetted version of the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def subset_samples_to_labels(self,
                                 labels: List[str],
                                 seed: int = 42,
                                 ) -> "LabeledDataset":
        """
        Keep only the samples corresponding to the labels passed in.
        Stacks with other labels; if `subset_samples_for_label`

        Arguments
        ---------
        labels: The label or labels of samples to keep
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        self: The subsetted version of the dataset
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


class RefineBioLabeledDataset(LabeledDataset):
    """ A dataset designed to store labeled data from a refine.bio compendium """
    def __init__(self,
                 compendium_path: Union[str, Path],
                 label_path: Union[str, Path],
                 metadata_path: Union[str, Path],
                 ) -> None:
        """
        A function to initialize the dataset from a compendium and lable mapping

        Arguments
        ---------
        compendium_path: The path to the compendium of expression data
        label_path: The path to the labels for the samples in the compendium
        metadata_path: The path to a file containing metadata for the samples
        """
        self.metadata = utils.parse_metadata_file(metadata_path)
        self.all_expression = utils.load_compendium_file(compendium_path)
        self.sample_to_label = utils.parse_label_file(label_path)
        self.sample_to_study = utils.map_sample_to_study(self.metadata,
                                                         list(self.all_expression.columns)
                                                         )

        # We will handle subsetting by creating views of the full dataset.
        # Most functions will access current_expression instead of all_expression.
        self.current_expression = self.all_expression

    def __len__(self) -> int:
        """
        Return the length of the dataset for use in determining
        the size of an epoch

        Returns
        -------
        length: The number of samples currently available in the dataset
        """
        return len(self.current_expression.index)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """
        Allows access into the dataset to select a single datapoint and its associated label

        Arguments
        ---------
        idx: The index of the given item

        Returns
        -------
        sample: The gene expression data for the given index in a genes x 1 array
        label: The label corresponding to the sample in X
        """
        sample = self.current_expression.iloc[:, idx].values
        label = np.array(self.sample_to_label[sample.index])
        return sample, label

    def get_all_data(self) -> Tuple[np.array, np.array]:
        """
        Returns all the expression data and labels from the dataset in the
        form of an (X,y) tuple where both X and y are numpy arrays

        Returns
        -------
        X: The gene expression data in a genes x samples array
        y: The label corresponding to each sample in X
        """

        X = self.current_expression.values
        sample_ids = self.current_expression.index
        labels = [self.sample_to_label[sample] for sample in sample_ids]
        y = np.array(labels)

        return X, y

    def subset_samples(self, fraction: float, seed: int = 42) -> "ExpressionDataset":
        """
        Limit the amount of data available to be returned in proportion to
        the total amount of data that was present in the dataset.
        That is to say that calling subset_data_by_fraction twice will not cause the Dataset
        object to have multiplicatively less data, it will have the fraction of data specified
        in the second call

        Arguments
        ---------
        fraction: The fraction of the samples to keep
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        self: The object after subsetting
        """
        samples_to_keep = int(len(self.all_expression.index) * fraction)
        self.current_expression = self.all_expression.sample(samples_to_keep,
                                                             axis='rows',
                                                             random_state=seed,
                                                             )

        return self

    def get_studies(self) -> Set[str]:
        """
        Return a list of study identifiers that contains all samples in
        current_expression

        Returns
        -------
        studies: The set of study identifiers
        """
        if (self.data_changed is None
                or self.data_changed
                or self.studies is None):

            self.data_changed = False
            samples = self.current_expression.columns

            studies = set()
            for sample in samples:
                try:
                    studies.add(self.sample_to_study[sample])
                except KeyError:
                    pass

            self.studies = studies

            return self.studies
        else:
            return self.studies

    def subset_studies(self,
                       fraction: float = None,
                       num_studies: int = None,
                       seed: int = 42) -> "ExpressionDataset":
        """
        This method is similar to `subset_samples`, but removes entire studies until the
        fraction of data remaining is less than the fraction passed in.

        Either `fraction` or `num_studies` must be specified. If both are specified,
        `num_studies` will be given preference.

        Arguments
        ---------
        fraction: The fraction of the samples to keep
        num_studies: The number of studies to keep
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        self: The object after subsetting

        Raises
        ------
        ValueError: Neither fraction nor num_studies were specified
        """
        random.seed(seed)

        studies = self.get_studies()
        samples = self.all_expression.columns

        if fraction is None:
            if num_studies is None:
                raise ValueError("Either fraction or num_studies must have a value")
            # Subset by number of studies
            else:
                studies_to_keep = random.sample(studies, num_studies)
                samples_to_keep = [sample for sample in samples
                                   if self.sample_to_study[sample] in studies_to_keep
                                   ]
                self.current_expression = self.all_expression.loc[:, samples_to_keep]

        # Subset by fraction
        else:
            total_samples = len(self.all_expression.columns)
            samples_to_keep = []
            shuffled_studies = random.sample(studies, len(studies))

            for study in shuffled_studies:
                if len(samples_to_keep) < fraction * total_samples:
                    break
                studies_to_keep.append(study)
                samples_to_keep = [sample for sample in samples
                                   if self.sample_to_study[sample] in studies_to_keep
                                   ]

        self.data_changed = True
        return self

    # For more info on using a forward reference for the type, see
    # https://github.com/python/mypy/issues/3661#issuecomment-313157498
    def get_cv_splits(self, num_splits: int, seed: int = 42) -> Sequence["ExpressionDataset"]:
        """
        Split the dataset into a list of smaller dataset objects with a roughly equal
        number of samples in each.

        If multiple studies are present in the dataset, each study should only
        be present in a single fold

        Arguments
        ---------
        num_splits: The number of groups to split the dataset into
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        subsets: A list of datasets, each composed of fractions of the original
        """
        raise NotImplementedError

    def train_test_split(self,
                         train_fraction: float = None,
                         train_study_count: int = None,
                         seed: int = 42,
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
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        train: The dataset with around the amount of data specified by train_fraction
        test: The dataset with the remaining data
        """
        raise NotImplementedError

    def subset_samples_for_label(self, fraction: float,
                                 label: str,
                                 seed: int = 42,
                                 ) -> "LabeledDataset":
        """
        Limit the number of samples available for a single label.
        For example, if you wanted to use only ten percent of the sepsis expression
        samples across all studies, you would use this function.

        Arguments
        ---------
        fraction: The fraction of the samples to keep
        label: The category of data to apply this subset to
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        self: The subsetted version of the dataset
        """
        self.data_changed = True
        raise NotImplementedError

    def subset_samples_to_labels(self,
                                 labels: List[str],
                                 seed: int = 42,
                                 ) -> "LabeledDataset":
        """
        Keep only the samples corresponding to the labels passed in.
        Stacks with other labels; if `subset_samples_for_label`

        Arguments
        ---------
        labels: The label or labels of samples to keep
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        self: The subsetted version of the dataset
        """
        self.data_changed = True
        raise NotImplementedError
