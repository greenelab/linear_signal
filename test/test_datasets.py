""" This file contains test cases for datasets.py
As new datasets are added, all that needs to be done is to create a function that
initializes it, like `create_refinebio_labeled_dataset`, and have the function added
to either the labeled_datasets or unlabeled_datasets fixture, depending on which
base class it inherited from
"""

import os

import numpy as np
import pytest

from saged import datasets


@pytest.fixture(scope="module")
def labeled_datasets():
    dataset_list = []
    dataset_list.append(create_refinebio_labeled_dataset())

    return dataset_list


@pytest.fixture(scope="module")
def unlabeled_datasets():
    labeled_dataset = create_refinebio_labeled_dataset()
    converted_dataset = datasets.RefineBioUnlabeledDataset.from_labeled_dataset(labeled_dataset)

    dataset_list = []
    dataset_list.append(create_refinebio_unlabeled_dataset())
    dataset_list.append(converted_dataset)

    return dataset_list


@pytest.fixture(scope="module")
def all_datasets(labeled_datasets, unlabeled_datasets):
    return labeled_datasets + unlabeled_datasets


def create_refinebio_labeled_dataset():
    """ Create a refinebio labeled dataset from test data """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    expression_path = os.path.join(test_dir, 'data', 'test_expression.tsv')
    label_path = os.path.join(test_dir, 'data', 'test_labels.pkl')
    metadata_path = os.path.join(test_dir, 'data', 'test_metadata.json')

    dataset = datasets.RefineBioLabeledDataset.from_paths(expression_path,
                                                          label_path,
                                                          metadata_path)

    return dataset


def create_refinebio_unlabeled_dataset():
    """ Create an unlabeled dataset from test data """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    expression_path = os.path.join(test_dir, 'data', 'test_expression.tsv')
    metadata_path = os.path.join(test_dir, 'data', 'test_metadata.json')

    dataset = datasets.RefineBioUnlabeledDataset.from_paths(expression_path,
                                                            metadata_path)

    return dataset


def test_get_item(labeled_datasets, unlabeled_datasets):
    for dataset in labeled_datasets:
        x, y = dataset[0]
        assert type(x) is np.ndarray
        assert type(y) is np.ndarray
    for dataset in unlabeled_datasets:
        x = dataset[0]
        assert type(x) is np.ndarray


def test_get_all_data(labeled_datasets, unlabeled_datasets):
    for dataset in labeled_datasets:
        X, y = dataset.get_all_data()
        assert type(X) is np.ndarray
        assert type(y) is np.ndarray
        assert X.shape[1] == y.shape[0]

    for dataset in unlabeled_datasets:
        assert type(X) is np.ndarray


@pytest.mark.parametrize("fraction", [.1, .5, .9])
def test_subset_samples(all_datasets, fraction):
    for dataset in all_datasets:
        total_samples = len(dataset)
        print(total_samples)

        subset_dataset = dataset.subset_samples(fraction, seed=1)
        subset_samples = subset_dataset.get_samples()
        assert len(subset_dataset) == int(total_samples * fraction)
        dataset.reset_filters()

        subset_dataset = dataset.subset_samples(fraction, seed=2)
        new_subset = subset_dataset.get_samples()
        print(id(subset_samples), id(new_subset))
        assert subset_samples != new_subset  # Make sure randomization works
        dataset.reset_filters()

        subset_dataset = dataset.subset_samples(fraction, seed=1)
        assert subset_samples == subset_dataset.get_samples()  # Make sure the seed works
        dataset.reset_filters()


def test_get_studies():
    dataset = create_refinebio_labeled_dataset()
    # We'll run the test case twice. The second run makes sure the cache code works correctly
    for i in range(2):
        studies = dataset.get_studies()
        assert 'study1' in studies
        assert 'study2' in studies
        assert 'study3' in studies
        assert 'study4' in studies
        assert 'study5' in studies
        assert 'study6' in studies


@pytest.mark.parametrize("fraction,num_studies",
                         [(.1, None),
                          (.5, None),
                          (.9, None),
                          (None, 1),
                          (None, 3),
                          (None, 5),
                          (.01, 5),  # Test that num_studies is given preference
                          (None, None)
                          ])
def test_subset_studies(all_datasets, fraction, num_studies):
    for dataset in all_datasets:
        if num_studies is None and fraction is None:
            with pytest.raises(ValueError):
                dataset.subset_studies(fraction, num_studies, seed=1)
            continue

        samples = None
        if num_studies is not None:
            subset_dataset = dataset.subset_studies(fraction, num_studies, seed=1)

            samples = subset_dataset.get_samples()
            studies = subset_dataset.get_studies()

            assert len(studies) == num_studies

        else:
            all_samples = dataset.get_samples()

            subset_dataset = dataset.subset_studies(fraction, num_studies, seed=1)
            samples = subset_dataset.get_samples()

            assert len(samples) > len(all_samples) * fraction

        dataset.reset_filters()

        # There are only six studies, so tests will spuriously fail when selecting many studies
        if fraction == .1 or num_studies == 1:
            # Ensure randomization works
            subset_dataset = dataset.subset_studies(fraction, num_studies, seed=2)
            assert samples != subset_dataset.get_samples()
            dataset.reset_filters()

            # Ensure setting random seed works
            subset_dataset = dataset.subset_studies(fraction, num_studies, seed=1)
            assert samples == subset_dataset.get_samples()
            dataset.reset_filters()


@pytest.mark.parametrize('num_splits', [1, 2, 6])
def test_get_cv_splits(all_datasets, num_splits):
    seed = 42
    for dataset in all_datasets:
        original_studies = set(dataset.get_studies())
        original_samples = set(dataset.get_samples())

        cv_studies = set()
        cv_samples = set()
        cv_datasets = dataset.get_cv_splits(num_splits, seed)

        for cv_dataset in cv_datasets:
            current_samples = cv_dataset.get_samples()
            current_studies = cv_dataset.get_studies()

            # No samples should be shared between splits
            for sample in current_samples:
                assert sample not in cv_samples
                cv_samples.add(sample)

            # No studies should be shared between splits
            for study in current_studies:
                assert study not in cv_studies
                cv_studies.add(study)

        # All studies should be accounted for in the spilts
        assert cv_studies == original_studies
        # All samples should be accounted for in the splits
        assert cv_samples == original_samples


@pytest.mark.parametrize('num_splits', [6])
def test_get_cv_randomness(all_datasets, num_splits):
    for dataset in all_datasets:
        cv_datasets = dataset.get_cv_splits(num_splits, seed=1)
        different_cv_datasets = dataset.get_cv_splits(num_splits, seed=2)
        same_cv_datasets = dataset.get_cv_splits(num_splits, seed=1)

        different_works = False
        for d1, d2, d3 in zip(cv_datasets, different_cv_datasets, same_cv_datasets):
            assert d1.get_studies() == d3.get_studies()
            if d1.get_studies() != d2.get_studies():
                different_works = True

        assert different_works


@pytest.mark.parametrize("train_fraction,train_study_count",
                         [(.1, None),
                          (.5, None),
                          (.9, None),
                          (None, 1),
                          (None, 3),
                          (None, 5),
                          (.01, 5),  # Test that train_study_count is given preference
                          (None, None)
                          ])
def test_train_test_split(all_datasets, train_fraction, train_study_count):
    seed = 42
    for dataset in all_datasets:
        original_studies = set(dataset.get_studies())
        original_samples = set(dataset.get_samples())

        # Function should throw a value error if train_fraction and train_study_count
        # are unspecified
        if train_fraction is None and train_study_count is None:
            with pytest.raises(ValueError):
                dataset.train_test_split(train_fraction, train_study_count, seed)
            continue

        train_data, test_data = dataset.train_test_split(train_fraction,
                                                         train_study_count,
                                                         seed)
        train_sample_list = train_data.get_samples()
        train_study_list = train_data.get_studies()

        test_sample_list = test_data.get_samples()
        test_study_list = test_data.get_studies()

        train_samples = set(train_sample_list)
        train_studies = set(train_study_list)
        test_samples = set(test_sample_list)
        test_studies = set(test_study_list)

        # Test for duplication in the returns
        assert len(train_sample_list) == len(train_samples)
        assert len(test_sample_list) == len(test_samples)
        assert len(train_study_list) == len(train_studies)
        assert len(test_study_list) == len(test_studies)

        # Make sure samples and studies aren't shared between splits
        assert len(train_samples & test_samples) == 0
        assert len(train_studies & test_studies) == 0

        # Test splitting by train study count
        if train_study_count is not None:
            assert len(train_studies) == train_study_count
            assert len(test_studies) == len(original_studies) - train_study_count

        # Test splitting by train sample fraction
        else:
            assert len(train_samples) >= len(original_samples) * train_fraction
            assert len(test_samples) <= len(original_samples) * (1-train_fraction)


@pytest.mark.parametrize('fraction, subset_label',
                         [(.1, 'label1'),
                          (.5, 'label1'),
                          (.9, 'label1'),
                          (.1, 'label2'),
                          (.5, 'label2'),
                          (.9, 'label2'),
                          (.1, 'label5'),
                          (.5, 'label5'),
                          (.9, 'label5'),
                          (0, 'label6'),
                          (1, 'label6'),
                          ])
def test_subset_samples_for_label(labeled_datasets, fraction, subset_label):
    for dataset in labeled_datasets:
        dataset.reset_filters()
        all_samples = dataset.get_samples()
        sample_to_label = dataset.sample_to_label

        dataset = dataset.subset_samples_for_label(fraction, subset_label, seed=1)
        subset_samples = dataset.get_samples()

        labels = set([label for label in sample_to_label.values()])

        all_sample_counts = {label: 0 for label in labels}
        subset_sample_counts = {label: 0 for label in labels}

        for sample in all_samples:
            all_sample_counts[sample_to_label[sample]] += 1

        for sample in subset_samples:
            subset_sample_counts[sample_to_label[sample]] += 1

        for label in all_sample_counts:
            if label == subset_label:
                assert subset_sample_counts[label] == int(all_sample_counts[label] * fraction)
            else:
                assert all_sample_counts[label] == subset_sample_counts[label]

        dataset.reset_filters()

        # Don't test randomization on groups that are too small
        if subset_sample_counts[subset_label] < 10:
            continue

        # Make sure randomization works
        dataset = dataset.subset_samples_for_label(fraction, subset_label, seed=2)
        assert subset_samples != dataset.get_samples()
        dataset.reset_filters()

        # Make sure setting a seed works
        dataset = dataset.subset_samples_for_label(fraction, subset_label, seed=1)
        assert subset_samples == dataset.get_samples()
        dataset.reset_filters()


@pytest.mark.parametrize('labels',
                         [['label1'],
                          ['label2', 'label4'],
                          [],
                          ['label1', 'label2', 'label3', 'label4', 'label5', 'label6'],
                          ['label5', 'label5']
                          ])
def test_subset_samples_to_labels(labeled_datasets, labels):
    for dataset in labeled_datasets:
        dataset = dataset.subset_samples_to_labels(labels)
        samples = dataset.get_samples()
        for sample in samples:
            assert dataset.sample_to_label[sample] in labels
        dataset.reset_filters()
