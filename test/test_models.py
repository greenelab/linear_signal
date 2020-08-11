""" Test the functions implemented in models.py """
import os

import numpy as np
import pytest
import yaml

import test_datasets
from saged import models, datasets

N_COMPONENTS = 2


@pytest.fixture(scope="module")
def sklearn_models():
    model_list = []
    model_list.append(models.LogisticRegression(seed=42))

    return model_list


@pytest.fixture(scope="module")
def unsupervised_models():
    model_list = []
    model_list.append(create_PCA())

    return model_list


@pytest.fixture(scope="module")
def pytorch_configs():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(test_dir, 'data')

    configs = []

    config_file_path = os.path.join(config_dir, 'three_layer_config.yml')
    with open(config_file_path) as config_file:
        configs.append(yaml.safe_load(config_file))

    return configs


@pytest.fixture(scope="module")
def pytorch_models(dataset, pytorch_configs):
    input_size = len(dataset.get_features())
    output_size = len(dataset.get_classes())

    model_list = []
    for config in pytorch_configs:
        config['input_size'] = input_size
        config['output_size'] = output_size
        model_class = getattr(models, config['name'])
        model = model_class(**config)
        model_list.append(model)

    return model_list


def create_PCA():
    seed = 42

    model = models.PCA(N_COMPONENTS, seed)
    return model


@pytest.fixture(scope="module")
def dataset():
    return test_datasets.create_refinebio_labeled_dataset()


@pytest.fixture(scope="module")
def unlabeled_dataset(dataset):
    unlabeled_dataset = datasets.RefineBioUnlabeledDataset.from_labeled_dataset(dataset)
    return unlabeled_dataset


def test_sklearn_fit_predict(sklearn_models,
                             dataset,
                             unlabeled_dataset):
    for model in sklearn_models:
        model = model.fit(dataset)

        predictions = model.predict(unlabeled_dataset)
        assert type(predictions) == np.ndarray


def test_sklearn_save_load_model(sklearn_models,
                                 dataset,
                                 unlabeled_dataset):
    for model in sklearn_models:
        model.fit(dataset)

        predictions = model.predict(unlabeled_dataset)

        model.save_model('model_test.pkl')
        loaded_model = type(model).load_model('model_test.pkl')

        new_predictions = loaded_model.predict(unlabeled_dataset)

        assert np.array_equal(predictions, new_predictions)

        os.remove('model_test.pkl')


def test_pytorch_save_load_model(pytorch_models,
                                 pytorch_configs,
                                 dataset,
                                 unlabeled_dataset):
    for model, config in zip(pytorch_models, pytorch_configs):
        model.fit(dataset)

        predictions = model.predict(unlabeled_dataset)

        model.save_model('model_test.pkl')

        loaded_model = type(model).load_model('model_test.pkl',
                                              **config,
                                              )
        # Send model to gpu if applicable
        loaded_model.model.to(loaded_model.device)

        new_predictions = loaded_model.predict(unlabeled_dataset)

        assert np.array_equal(predictions.cpu(), new_predictions.cpu())

        os.remove('model_test.pkl')


def test_load_params(pytorch_models, dataset):
    for model in pytorch_models:
        original_params = model.get_parameters()

        model = model.fit(dataset)
        new_params = model.get_parameters()

        diff = False
        for orig_key, trained_key in zip(original_params, new_params):
            if not np.array_equal(original_params[orig_key].cpu(),
                                  new_params[trained_key].cpu()):
                diff = True

        assert diff is True

        model = model.load_parameters(original_params)
        loaded_params = model.get_parameters()

        for orig_key, loaded_key in zip(original_params, loaded_params):
            assert np.array_equal(original_params[orig_key].cpu(),
                                  loaded_params[loaded_key].cpu())


def test_pytorch_fit_predict(pytorch_models, dataset):
    for model in pytorch_models:
        original_params = model.get_parameters()

        model = model.fit(dataset)
        trained_params = model.get_parameters()

        diff = False
        for orig_key, trained_key in zip(original_params, trained_params):
            if not np.array_equal(original_params[orig_key].cpu(),
                                  trained_params[trained_key].cpu()):
                diff = True
        assert diff is True


def test_unsupervised_fit_transform(unsupervised_models, unlabeled_dataset):
    for model in unsupervised_models:
        X = unlabeled_dataset.get_all_data()

        embedded_data = model.fit_transform(unlabeled_dataset)

        embedded_X = embedded_data.get_all_data()

        # Ensure the number of samples is maintained
        assert X.shape[0] == embedded_X.shape[0]
        # Ensure dimension was reduced
        assert embedded_X.shape[1] == N_COMPONENTS

        # Ensure sample ids are preserved
        assert embedded_data.get_samples() == unlabeled_dataset.get_samples()


def test_pca_save_load(unlabeled_dataset):
    model = create_PCA()

    embedded_data = model.fit_transform(unlabeled_dataset)

    model.save_model('model_test.pkl')
    loaded_model = type(model).load_model('model_test.pkl')

    new_embedded = loaded_model.transform(unlabeled_dataset)

    assert np.array_equal(embedded_data, new_embedded)

    os.remove('model_test.pkl')


def test_evaluate(pytorch_models, dataset):
    for model in pytorch_models:
        preds, true_labels = model.evaluate(dataset)

        assert type(preds) == np.ndarray
        assert type(true_labels) == np.ndarray
