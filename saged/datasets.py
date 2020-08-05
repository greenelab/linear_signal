""" This file contains dataset objects for manipulating gene expression data """
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Tuple, List, Union, Set, Dict

import numpy as np
import pandas as pd
from sklearn import preprocessing

from saged import utils


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
    @classmethod
    def from_config(class_object,
                    ) -> "RefineBioUnlabeledDataset":
        """
        A function to initialize a RefineBioUnlabeledDataset object given a config dict

        Returns
        -------
        new_dataset: The initialized dataset
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_data(self) -> Tuple[np.array, np.array]:
        """
        Returns all the expression data and labels from the dataset in the
        form of an (X,y) tuple where both X and y are numpy arrays

        Returns
        -------
        X: The gene expression data in a samples x genes array
        y: The label corresponding to each sample in X
        """
        raise NotImplementedError

    @abstractmethod
    def set_all_data(self, new_data: np.array) -> None:
        """
        Overwrite the current data stored in the dataset with the new one passed in
        """
        raise NotImplementedError

    @abstractmethod
    def get_features(self) -> List[str]:
        """
        Return the list of the ids of all features in the dataset. Usually these will be genes,
        but may be PCs or other latent dimensions if the dataset is transformed

        Returns
        -------
        features: The features ids for all features in the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def get_samples(self) -> List[str]:
        """
        Return the sample ids for all samples in the dataset

        Returns
        -------
        samples: The list of sample ids
        """
        raise NotImplementedError

    @abstractmethod
    def reset_filters(self) -> None:
        """
        Restore dataset to its original state, reversing any subsetting operations
        performed upon it
        """
        raise NotImplementedError

    @abstractmethod
    def subset_samples(self, fraction: float, seed: int = 42) -> "ExpressionDataset":
        """
        Limit the amount of data available to be returned in proportion to
        the amount of data that was available in the dataset

        Arguments
        ---------
        fraction: The fraction of the samples to keep
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        self: The object after subsetting
        """
        raise NotImplementedError

    @abstractmethod
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
                                 ) -> "LabeledDataset":
        """
        Keep only the samples corresponding to the labels passed in.
        Stacks with other subset calls

        Arguments
        ---------
        labels: The label or labels of samples to keep

        Returns
        -------
        self: The subsetted version of the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def get_classes(self) -> Set[str]:
        """
        Return the set of all class labels in the current dataset

        Returns
        -------
        classes: The set of class labels
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

class MixedDataset(LabeledDataset):
    """ A dataset containing both labeled and unlabeled samples """
    @abstractmethod
    def get_labeled() -> LabeledDataset:
        """
        Return the labeled samples from the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def get_unlabeled() -> UnlabeledDataset:
        """
        Return the unlabeled samples from the dataset
        """
        raise NotImplementedError


class RefineBioDataset(ExpressionDataset):
    """
    A class containing logic used by all the types of RefineBio datasets.
    The RefineBio dataset inheritance pattern is to inherit both this class and the class
    denoting the type of dataset it is (LabeledDataset, UnlabeledDataset, or MixedDatset).
    """
    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    @classmethod
    def from_config():
        """
        A function to initialize a RefineBioDataset object given a config dict

        Returns
        -------
        new_dataset: The initialized dataset
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Return the length of the dataset for use in determining
        the size of an epoch

        Returns
        -------
        length: The number of samples currently available in the dataset
        """
        return len(self.current_expression.columns)

    def set_all_data(self, new_data: np.array) -> None:
        """
        Overwrite the current data stored in the dataset with the new one passed in

        Arguments
        ---------
        new_data: The variable x samples array to overwrite the old data with. The first dimension
                  can be anything, but the sample dimension must remain the same as in the old data
        """
        current_samples = self.get_samples()
        new_dataframe = pd.DataFrame(data=new_data, columns=current_samples)

        self.all_expression = new_dataframe
        self.current_expression = new_dataframe

    def reset_filters(self):
        """
        Restore the current_expression attribute to contain all expression data
        """
        self.current_expression = self.all_expression
        self.data_changed = True

    def subset_samples(self, fraction: float, seed: int = 42) -> "RefineBioDataset":
        """
        Limit the amount of data available to be returned in proportion to
        the amount of data that was available in the dataset

        Arguments
        ---------
        fraction: The fraction of the samples to keep
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        self: The object after subsetting
        """
        samples_to_keep = int(len(self) * fraction)
        self.current_expression = self.current_expression.sample(samples_to_keep,
                                                                 axis='columns',
                                                                 random_state=seed,
                                                                 )
        self.data_changed = True

        return self

    def get_samples(self) -> List[str]:
        """
        Return the list of sample accessions for all samples currently available in the dataset
        """
        return list(self.current_expression.columns)

    def get_features(self) -> List[str]:
        """
        Return the list of the ids of all features in the dataset. Usually these will be genes,
        but may be PCs or other latent dimensions if the dataset is transformed

        Returns
        -------
        features: The features ids for all features in the dataset
        """
        return list(self.current_expression.index)

    def get_studies(self) -> Set[str]:
        """
        Return a list of study identifiers that contains all samples in
        current_expression

        Returns
        -------
        studies: The set of study identifiers
        """
        # self.data_changed isn't set by the init function, so calling get_studies before any
        # subsetting needs to check whether the data_changed attribute exists
        if (not hasattr(self, 'data_changed')
                or self.data_changed
                or self.studies is None):

            self.data_changed = False
            samples = self.get_samples()

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
                       seed: int = 42) -> "RefineBioDataset":
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

        if fraction is None and num_studies is None:
            raise ValueError("Either fraction or num_studies must have a value")
        if num_studies is not None:
            # Subset by number of studies
            studies_to_keep = random.sample(studies, num_studies)
            samples_to_keep = utils.get_samples_in_studies(samples,
                                                           studies_to_keep,
                                                           self.sample_to_study)

            self.current_expression = self.all_expression.loc[:, samples_to_keep]

        # Subset by fraction
        else:
            total_samples = len(self.all_expression.columns)
            samples_to_keep = []
            shuffled_studies = random.sample(studies, len(studies))
            studies_to_keep = set()

            for study in shuffled_studies:
                if len(samples_to_keep) > fraction * total_samples:
                    break
                studies_to_keep.add(study)
                samples_to_keep = utils.get_samples_in_studies(samples,
                                                               studies_to_keep,
                                                               self.sample_to_study)

            self.current_expression = self.all_expression.loc[:, samples_to_keep]

        self.data_changed = True
        return self

    def get_cv_expression(self, num_splits: int, seed: int = 42) -> List[pd.DataFrame]:
        """
        Split the expression present in the dataset into `num_splits` partitions for
        use in cross-validation

        Arguments
        ---------
        num_splits: The number of groups to split the dataset into
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        cv_dataframes: The expression dataframe views for each cv split
        """
        random.seed(seed)

        samples = self.get_samples()
        studies = self.get_studies()
        shuffled_studies = random.sample(studies, len(studies))

        base_study_count = len(studies) // num_splits
        leftover = len(studies) % num_splits
        study_index = 0

        cv_dataframes = []
        for i in range(num_splits):
            study_count = base_study_count
            if i < leftover:
                study_count += 1

            current_studies = shuffled_studies[study_index:study_index+study_count]
            study_index += study_count

            current_samples = utils.get_samples_in_studies(samples,
                                                           current_studies,
                                                           self.sample_to_study)

            cv_expression = self.all_expression.loc[:, current_samples]
            cv_dataframes.append(cv_expression)

        return cv_dataframes

    def get_cv_splits(self, num_splits: int, seed: int = 42) -> Sequence["RefineBioDataset"]:
        """
        Split the dataset into a list of smaller dataset objects with a roughly equal
        number of studies in each.

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
        cv_dataframes = self.get_cv_expression(num_splits, seed)

        cv_datasets = []
        for expression_df in cv_dataframes:
            cv_dataset = RefineBioUnlabeledDataset(expression_df,
                                                   self.sample_to_study,
                                                   )
            cv_datasets.append(cv_dataset)

        return cv_datasets

    def get_train_test_expression(self,
                                  train_fraction: float = None,
                                  train_study_count: int = None,
                                  seed: int = 42,
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the expression into a train fraction and a test fraction.
        This function does the heavy lifting for train_test_split, but is abstracted
        out because different datasets will be formed at the end based on the calling class

        Arguments
        ---------
        train_fraction: The minimum fraction of the data to be used as training data.
            In reality, the fraction won't be met entirely due to the constraint of
            preserving studies.
        train_study_count: The number of studies to be included in the training set
        seed: The seed for the random number generator involved in subsetting

        Returns
        -------
        train_expression: The expression to be used in training
        test_expression: The expression to be used as a test set

        Raises
        ------
        ValueError if neither train_fraction nor train_study count are specified
        """
        random.seed(seed)

        studies = self.get_studies()
        samples = self.get_samples()
        shuffled_studies = random.sample(studies, len(studies))
        train_studies = []
        test_studies = []

        if train_fraction is None and train_study_count is None:
            raise ValueError("Either train_fraction or train_study_count must have a value")

        if train_study_count is not None:
            # Split by number of train studies
            train_studies = shuffled_studies[:train_study_count]
            test_studies = shuffled_studies[train_study_count:]

        # Subset by fraction
        else:
            total_samples = len(samples)
            train_samples = []
            samples_to_keep = []
            studies_to_keep = set()
            last_study_index = 0
            shuffled_studies = random.sample(studies, len(studies))

            for study in shuffled_studies:
                if len(samples_to_keep) > train_fraction * total_samples:
                    break
                studies_to_keep.add(study)

                # This is inefficient compared to adding samples for each study as the study is
                # added, but
                samples_to_keep = utils.get_samples_in_studies(samples,
                                                               studies_to_keep,
                                                               self.sample_to_study)
                last_study_index += 1

            train_studies = shuffled_studies[:last_study_index]
            test_studies = shuffled_studies[last_study_index:]

        train_samples = utils.get_samples_in_studies(samples,
                                                     train_studies,
                                                     self.sample_to_study)
        test_samples = utils.get_samples_in_studies(samples,
                                                    test_studies,
                                                    self.sample_to_study)

        train_expression = self.all_expression.loc[:, train_samples]
        test_expression = self.all_expression.loc[:, test_samples]

        return train_expression, test_expression


class RefineBioUnlabeledDataset(RefineBioDataset, UnlabeledDataset):
    """ A dataset containing data from a refine.bio compendium without labels """

    def __init__(self,
                 expression_df: pd.DataFrame,
                 sample_to_study: Dict[str, str]
                 ):
        """
        An initializer for the class. Typically this initializer will not be used directly,
        but will instead be called by `RefineBioUnlabeledDataset.from_paths()`

        Arguments
        ---------
        expression_df: The dataframe containing expression data where rows are genes and
            columns are samples
        sample_to_study: A mapping between sample accessions and their study accessions
        """
        self.all_expression = expression_df
        self.sample_to_study = sample_to_study

        # We will handle subsetting by creating views of the full dataset.
        # Most functions will access current_expression instead of all_expression.
        self.current_expression = expression_df

    @classmethod
    def from_config(class_object,
                    compendium_path: Union[str, Path],
                    metadata_path: Union[str, Path],
                    ) -> "RefineBioUnlabeledDataset":
        """
        A function to create a new object from paths to its data

        Arguments
        ---------
        compendium_path: The path to the compendium of expression data
        metadata_path: The path to a file containing metadata for the samples

        Returns
        -------
        new_dataset: The initialized dataset
        """
        metadata = utils.parse_metadata_file(metadata_path)
        all_expression = utils.load_compendium_file(compendium_path)
        sample_to_study = utils.map_sample_to_study(metadata,
                                                    list(all_expression.columns)
                                                    )

        new_dataset = class_object(all_expression, sample_to_study)
        return new_dataset

    @classmethod
    def from_list(class_object,
                  dataset_list: Sequence["RefineBioUnlabeledDataset"],
                  ) -> "RefineBioUnlabeledDataset":
        """
        Create a dataset by combining a list of other datasets

        Arguments
        ---------
        dataset_list: The list of datasets to combine

        Returns
        -------
        combined_dataset: The dataset created from the other datasets

        Raises
        ------
        ValueError: If no datasets are in the list
        """
        if len(dataset_list) == 0:
            raise ValueError('The provided list of datasets is empty')

        elif len(dataset_list) == 1:
            return dataset_list[0]

        else:
            combined_dataframe = dataset_list[0].current_expression

            for dataset in dataset_list[1:]:
                current_dataframe = dataset.current_expression

                # Ensure no sample gets duplicated
                # https://stackoverflow.com/questions/19125091/pandas-merge-how-to-avoid-duplicating-columns
                cols_to_use = current_dataframe.columns.difference(combined_dataframe.columns)

                # Combine dataframes
                combined_dataframe = pd.merge(combined_dataframe,
                                              current_dataframe[cols_to_use],
                                              left_index=True,
                                              right_index=True,
                                              how='outer')

            combined_dataset = RefineBioUnlabeledDataset(combined_dataframe)
            return combined_dataset

    @classmethod
    def from_labeled_dataset(class_object,
                             dataset: "RefineBioLabeledDataset") -> "RefineBioUnlabeledDataset":
        """
        Create a new unlabeled dataset instance from a labeled dataset

        Arguments
        ---------
        dataset: The labeled refinebio datase to be converted
        """
        new_dataset = class_object(dataset.all_expression, dataset.sample_to_study)
        return new_dataset

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """
        Allows access into the dataset to select a single datapoint

        Arguments
        ---------
        idx: The index of the given item

        Returns
        -------
        sample: The gene expression data for the given index in a genes x 1 array
        """
        sample_id = self.get_samples()[idx]
        sample = self.current_expression[sample_id].values
        return sample

    def get_all_data(self) -> Tuple[np.array, np.array]:
        """
        Returns all the expression data from the dataset in the
        form of numpy array

        Returns
        -------
        X: The gene expression data in a samples x genes array
        """
        X = self.current_expression.values.T

        return X

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

        Raises
        ------
        ValueError if neither train_fraction nor train_study count are specified
        """
        train_expression, test_expression = self.get_train_test_expression(train_fraction,
                                                                           train_study_count,
                                                                           seed,
                                                                           )

        train_dataset = RefineBioUnlabeledDataset(train_expression,
                                                  self.sample_to_study,
                                                  )
        test_dataset = RefineBioUnlabeledDataset(test_expression,
                                                 self.sample_to_study,
                                                 )
        return train_dataset, test_dataset


class RefineBioLabeledDataset(RefineBioDataset, RefineBioUnlabeledDataset):
    """ A dataset designed to store labeled data from a refine.bio compendium """
    def __init__(self,
                 expression_df: pd.DataFrame,
                 sample_to_label: Dict[str, str],
                 sample_to_study: Dict[str, str],
                 ):
        """
        An initializer for the RefineBioLabeledDataset. Typically this initializer
        will not be used directly, but will instead be called by
        RefineBioLabeledDataset.from_paths

        Arguments
        ---------
        expression_df: The dataframe containing expression data where rows are genes and
            columns are samples
        sample_to_label: A mapping between sample accessions and their phenotype labels
        sample_to_study: A mapping between sample accessions and their study accessions
        """
        self.all_expression = expression_df
        self.sample_to_label = sample_to_label
        self.sample_to_study = sample_to_study

        label_encoder = preprocessing.LabelEncoder()
        labels = [sample_to_label[sample] for sample in expression_df.columns]
        label_encoder.fit(labels)
        self.label_encoder = label_encoder

        self.current_expression = expression_df

    @classmethod
    def from_config(class_object,
                    compendium_path: Union[str, Path],
                    label_path: Union[str, Path],
                    metadata_path: Union[str, Path],
                    ) -> "RefineBioLabeledDataset":
        """
        Create a new dataset from paths to its data

        Arguments
        ---------
        compendium_path: The path to the compendium of expression data
        label_path: The path to the labels for the samples in the compendium
        metadata_path: The path to a file containing metadata for the samples

        Returns
        -------
        new_dataset: The initialized dataset
        """
        metadata = utils.parse_metadata_file(metadata_path)
        all_expression = utils.load_compendium_file(compendium_path)
        sample_to_label = utils.parse_label_file(label_path)
        sample_to_study = utils.map_sample_to_study(metadata,
                                                    list(all_expression.columns)
                                                    )

        new_dataset = class_object(all_expression, sample_to_label, sample_to_study)
        return new_dataset

    @classmethod
    def from_list(class_object,
                  dataset_list: Sequence["RefineBioLabeledDataset"],
                  ) -> "RefineBioLabeledDataset":
        """
        Create a dataset by combining a list of other datasets

        Arguments
        ---------
        dataset_list: The list of datasets to combine

        Returns
        -------
        combined_dataset: The dataset created from the other datasets

        Raises
        ------
        ValueError: If no datasets are in the list
        """
        if len(dataset_list) == 0:
            raise ValueError('The provided list of datasets is empty')

        elif len(dataset_list) == 1:
            return dataset_list[0]

        else:
            combined_dataframe = dataset_list[0].current_expression
            combined_labels = dataset_list[0].sample_to_label

            for dataset in dataset_list[1:]:
                current_dataframe = dataset.current_expression
                current_labels = dataset.sample_to_label

                # Ensure no sample gets duplicated
                # https://stackoverflow.com/questions/19125091/pandas-merge-how-to-avoid-duplicating-columns
                cols_to_use = current_dataframe.columns.difference(combined_dataframe.columns)

                # Combine dataframes
                combined_dataframe = pd.merge(combined_dataframe,
                                              current_dataframe[cols_to_use],
                                              left_index=True,
                                              right_index=True,
                                              how='outer')
                # Combine labels (double start expands a dictionary into key-value pairs)
                combined_labels = {**combined_labels, **current_labels}

            combined_dataset = RefineBioLabeledDataset(combined_dataframe, combined_labels)
            return combined_dataset

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
        sample_id = self.get_samples()[idx]
        sample = self.current_expression[sample_id].values
        label = self.sample_to_label[sample_id]
        encoded_label = self.label_encoder.transform([label])

        return sample, encoded_label

    def get_all_data(self) -> Tuple[np.array, np.array]:
        """
        Returns all the expression data and labels from the dataset in the
        form of an (X,y) tuple where both X and y are numpy arrays

        Returns
        -------
        X: The gene expression data in a samples x genes array
        y: The label corresponding to each sample in X
        """

        X = self.current_expression.values.T
        sample_ids = self.get_samples()
        labels = [self.sample_to_label[sample] for sample in sample_ids]
        y = self.label_encoder.transform(labels)

        return X, y

    def get_classes(self) -> Set[str]:
        """
        Return the set of all class labels in the current dataset

        Returns
        -------
        classes: The set of class labels
        """
        classes = set()
        for sample in self.get_samples():
            classes.add(self.sample_to_label[sample])

        return classes

    def recode(self) -> None:
        """
        Retrain the label encoder to contain only the labels present in current_expression instead
        of all the labels in the dataset
        """
        labels = [self.sample_to_label[sample] for sample in self.current_expression.columns]
        self.label_encoder.fit(labels)

    def map_labels_to_counts(self) -> Dict[str, int]:
        """
        Get the number of samples with each label

        Returns
        -------
        label_counts: A dictionary mapping labels to the number of samples with each label
        """
        label_counts = {}

        sample_ids = self.get_samples()
        labels = np.array([self.sample_to_label[sample] for sample in sample_ids])

        unique_elements, element_counts = np.unique(labels, return_counts=True)

        for label, count in zip(unique_elements, element_counts):
            label_counts[label] = count

        return label_counts

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
        random.seed(seed)

        all_samples = self.get_samples()
        samples_with_label = []
        samples_without_label = []

        for sample in all_samples:
            if self.sample_to_label[sample] == label:
                samples_with_label.append(sample)
            else:
                samples_without_label.append(sample)

        num_to_keep = int(len(samples_with_label) * fraction)

        samples_with_label_to_keep = random.sample(samples_with_label, num_to_keep)

        samples_to_keep = samples_without_label + samples_with_label_to_keep

        self.current_expression = self.current_expression.loc[:, samples_to_keep]

        return self

    def subset_samples_to_labels(self,
                                 labels: List[str],
                                 ) -> "LabeledDataset":
        """
        Keep only the samples corresponding to the labels passed in.
        Stacks with other subset calls

        Arguments
        ---------
        labels: The label or labels of samples to keep

        Returns
        -------
        self: The subsetted version of the dataset
        """
        self.data_changed = True

        label_set = set(labels)
        all_samples = self.get_samples()

        samples_to_keep = [sample for sample in all_samples
                           if self.sample_to_label[sample] in label_set]

        self.current_expression = self.current_expression.loc[:, samples_to_keep]

        return self

    def get_cv_splits(self, num_splits: int, seed: int = 42) -> Sequence["ExpressionDataset"]:
        """
        Split the dataset into a list of smaller dataset objects with a roughly equal
        number of studies in each.

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
        cv_dataframes = self.get_cv_expression(num_splits, seed)

        cv_datasets = []
        for expression_df in cv_dataframes:
            cv_dataset = RefineBioLabeledDataset(expression_df,
                                                 self.sample_to_label,
                                                 self.sample_to_study,
                                                 )
            cv_datasets.append(cv_dataset)

        return cv_datasets

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

        Raises
        ------
        ValueError if neither train_fraction nor train_study count are specified
        """
        train_expression, test_expression = self.get_train_test_expression(train_fraction,
                                                                           train_study_count,
                                                                           seed,
                                                                           )

        train_dataset = RefineBioLabeledDataset(train_expression,
                                                self.sample_to_label,
                                                self.sample_to_study,
                                                )
        test_dataset = RefineBioLabeledDataset(test_expression,
                                               self.sample_to_label,
                                               self.sample_to_study,
                                               )
        return train_dataset, test_dataset


class RefineBioMixedDataset(RefineBioDataset, MixedDataset):
    """
    A class to contain both labeled and unlabeled data. This class should be able to
    work with unsupervised models, but besides that should contain less functionality than
    in RefineBioLabeledDataset
    """

    def __init__(self,
                 expression_df: pd.DataFrame,
                 sample_to_label: Dict[str, str],
                 sample_to_study: Dict[str, str],
                 ):
        """
        An initializer for the RefineBioMixedDataset. Typically this initializer
        will not be used directly, but will instead be called by
        RefineBioMixedDataset.from_config

        Arguments
        ---------
        expression_df: The dataframe containing expression data where rows are genes and
            columns are samples
        sample_to_label: A mapping between sample accessions and their phenotype labels
        sample_to_study: A mapping between sample accessions and their study accessions
        """
        self.all_expression = expression_df
        self.sample_to_label = sample_to_label
        self.sample_to_study = sample_to_study

        label_encoder = preprocessing.LabelEncoder()
        labels = [sample_to_label[sample] for sample in expression_df.columns]
        label_encoder.fit(labels)
        self.label_encoder = label_encoder

        self.current_expression = expression_df

    @classmethod
    def from_config(class_object,
                    compendium_path: Union[str, Path],
                    label_path: Union[str, Path],
                    metadata_path: Union[str, Path],
                    ) -> "RefineBioLabeledDataset":
        """
        Create a new dataset from paths to its data

        Arguments
        ---------
        compendium_path: The path to the compendium of expression data
        label_path: The path to the labels for the samples in the compendium
        metadata_path: The path to a file containing metadata for the samples

        Returns
        -------
        new_dataset: The initialized dataset
        """
        metadata = utils.parse_metadata_file(metadata_path)
        all_expression = utils.load_compendium_file(compendium_path)
        sample_to_label = utils.parse_label_file(label_path)
        sample_to_study = utils.map_sample_to_study(metadata,
                                                    list(all_expression.columns)
                                                    )

        new_dataset = class_object(all_expression, sample_to_label, sample_to_study)
        return new_dataset

    def __len__(self) -> int:
        """
        Return the length of the dataset for use in determining
        the size of an epoch

        Returns
        -------
        length: The number of samples currently available in the dataset
        """
        return len(self.current_expression.columns)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """
        Allows access into the dataset to select a single datapoint

        Arguments
        ---------
        idx: The index of the given item

        Returns
        -------
        sample: The gene expression data for the given index in a genes x 1 array
        """
        sample_id = self.get_samples()[idx]
        sample = self.current_expression[sample_id].values
        return sample

    def get_all_data(self) -> Tuple[np.array, np.array]:
        """
        Returns all the expression data from the dataset in the
        form of numpy array

        Returns
        -------
        X: The gene expression data in a samples x genes array
        """
        X = self.current_expression.values.T

        return X

    def get_labeled(self) -> RefineBioLabeledDataset:
        """
        Return the labeled samples from the dataset
        """
        samples = self.get_samples()
        labeled_samples = [sample for sample in samples if sample in self.sample_to_label]

        labeled_expression = self.current_expression.loc[:, labeled_samples]

        new_dataset = RefineBioLabeledDataset(labeled_expression,
                                              self.sample_to_label,
                                              self.sample_to_study,
                                              )

        return new_dataset

    def get_unlabeled(self) -> RefineBioUnlabeledDataset:
        """
        Return the unlabeled samples from the dataset
        """
        samples = self.get_samples()
        unlabeled_samples = [sample for sample in samples if sample not in self.sample_to_label]

        unlabeled_expression = self.current_expression.loc[:, unlabeled_samples]

        new_dataset = RefineBioLabeledDataset(unlabeled_expression,
                                              self.sample_to_study,
                                              )

        return new_dataset
