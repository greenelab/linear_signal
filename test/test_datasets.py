""" This file contains test cases for datasets.py
As new datasets are added, all that needs to be done is to create a function that
initializes it, like `create_refinebio_labeled_dataset`, and have the function added
to either the labeled_datasets or unlabeled_datasets fixture, depending on which
base class it inherited from
"""

import os
import random

import numpy as np
import pytest
import yaml

from saged import datasets, generate_test_data


@pytest.fixture(scope="function")
def labeled_datasets():
    dataset_list = []
    dataset_list.append(create_refinebio_labeled_dataset())

    return dataset_list


@pytest.fixture(scope="function")
def unlabeled_datasets():
    labeled_dataset = create_refinebio_labeled_dataset()
    converted_dataset = datasets.RefineBioUnlabeledDataset.from_labeled_dataset(labeled_dataset)

    dataset_list = []
    dataset_list.append(create_refinebio_unlabeled_dataset())
    dataset_list.append(converted_dataset)

    return dataset_list


@pytest.fixture(scope='function')
def mixed_datasets():
    dataset_list = []
    dataset_list.append(create_refinebio_mixed_dataset())

    return dataset_list


@pytest.fixture(scope="function")
def all_datasets(labeled_datasets, unlabeled_datasets, mixed_datasets):
    return labeled_datasets + unlabeled_datasets + mixed_datasets


def create_refinebio_labeled_dataset():
    """ Create a refinebio labeled dataset from test data """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(test_dir, 'data', 'test_data_config.yml')
    with open(config_file_path) as config_file:
        config = yaml.safe_load(config_file)

    dataset = datasets.RefineBioLabeledDataset.from_config(**config)
    return dataset


def create_refinebio_unlabeled_dataset():
    """ Create an unlabeled dataset from test data """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(test_dir, 'data', 'test_data_config.yml')
    with open(config_file_path) as config_file:
        config = yaml.safe_load(config_file)

    # We could write a config file without the labels, but this is easier
    del(config['label_path'])

    dataset = datasets.RefineBioUnlabeledDataset.from_config(**config)

    return dataset


def create_refinebio_mixed_dataset():
    """ Create a dataset with both labeled and unlabeled data """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(test_dir, 'data', 'test_mixed_data_config.yml')
    with open(config_file_path) as config_file:
        config = yaml.safe_load(config_file)

    dataset = datasets.RefineBioMixedDataset.from_config(**config)
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
        assert X.shape[0] == y.shape[0]

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


@pytest.mark.parametrize("samples",
                         [(['GSM1368585', 'SRR4427914', 'GSM753652']),
                          (['ERR1275178']),
                          (['GSM1692420']),
                          ])
def test_subset_to_samples(all_datasets, samples):
    for dataset in all_datasets:
        dataset.subset_to_samples(samples)

        subset_samples = dataset.get_samples()
        assert set(subset_samples) == set(samples)
        dataset.reset_filters()


def test_subset_to_samples_raises_keyerror(all_datasets):
    samples = ['ThisSampleDoesNotExist', 'SRR4427914']
    for dataset in all_datasets:
        with pytest.raises(KeyError):
            dataset.subset_to_samples(samples)


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


def test_set_all_data(all_datasets):
    for dataset in all_datasets:
        current_data = dataset.get_all_data()

        if issubclass(type(dataset), datasets.LabeledDataset):
            # Throw out labels in a labeled dataset
            current_data = current_data[0]

        new_data = np.random.normal(size=(1337, len(dataset.get_samples())))
        print(new_data.shape)

        dataset.set_all_data(new_data)

        assert not np.array_equal(new_data, current_data)

        retrieved_new_data = dataset.get_all_data()

        if issubclass(type(dataset), datasets.LabeledDataset):
            # Throw out labels in a labeled dataset
            retrieved_new_data = retrieved_new_data[0]

        assert np.array_equal(new_data, retrieved_new_data.T)


def test_get_features(all_datasets):
    for dataset in all_datasets:
        features = dataset.get_features()

        # Generate the gene names as in generate_test_data
        gene_names = []
        for i in range(generate_test_data.NUM_GENES):
            gene_base = 'ENSG0000000000'
            gene_name = '{}{}'.format(gene_base, i)
            gene_names.append(gene_name)
        assert features == gene_names


def test_get_samples(all_datasets):
    for dataset in all_datasets:
        samples = dataset.get_samples()

        random.seed(42)

        # Get sample names as in `generate_test_data.py`
        current_dir = os.path.dirname(os.path.abspath(__file__))
        compendium_path = os.path.join(current_dir, '../data/HOMO_SAPIENS.tsv')

        # Pull 200 random sample names from the compendium
        compendium_head = None
        with open(compendium_path, 'r') as compendium_file:
            compendium_head = compendium_file.readline().strip().split('\t')

        true_samples = random.sample(compendium_head, generate_test_data.NUM_SAMPLES)

        assert samples == true_samples


def test_get_labeled(mixed_datasets):
    for dataset in mixed_datasets:
        labeled_dataset = dataset.get_labeled()
        unlabeled_dataset = dataset.get_unlabeled()

        labeled_samples = set(labeled_dataset.get_samples())
        unlabeled_samples = set(unlabeled_dataset.get_samples())
        all_samples = set(dataset.get_samples())

        assert labeled_samples.issubset(all_samples)
        assert unlabeled_samples.issubset(all_samples)
        labeled_samples.update(unlabeled_samples)
        assert labeled_samples == all_samples

        labeled_features = set(labeled_dataset.get_features())
        unlabeled_features = set(unlabeled_dataset.get_features())
        all_features = set(dataset.get_features())

        assert labeled_features.issubset(all_features)
        assert unlabeled_features.issubset(all_features)
        labeled_features.update(unlabeled_features)
        assert labeled_features == all_features


def test_from_list(all_datasets):
    for dataset in all_datasets:
        dataset_list = dataset.get_cv_splits(num_splits=2, seed=42)

        reformed_dataset = type(dataset).from_list(dataset_list)

        original_samples = dataset.get_samples()
        reformed_samples = reformed_dataset.get_samples()
        assert set(original_samples) == set(reformed_samples)

        original_features = dataset.get_features()
        reformed_features = reformed_dataset.get_features()
        assert set(original_features) == set(reformed_features)
