""" Test the functions implemented in models.py """
import os

import numpy as np
import pytest
import torch
import yaml

import test_datasets
from saged import models, datasets

N_COMPONENTS = 2


@pytest.fixture(scope="module")
def sklearn_models():
    model_list = []
    model_list.append(models.LogisticRegression())

    return model_list


@pytest.fixture(scope="module")
def unsupervised_models():
    model_list = []
    model_list.append(create_PCA())

    return model_list


@pytest.fixture(scope="module")
def pytorch_models():
    model_list = []
    # Pytorch models go here
    model_list.append(create_three_layer_net())

    return model_list


@pytest.fixture(scope="module")
def config():
    config_dict = None

    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(test_dir, 'data', 'test_config.yml')
    with open(config_path) as config_file:
        config_dict = yaml.safe_load(config_file)

    return config_dict


def create_PCA():
    seed = 42

    model = models.PCA(N_COMPONENTS, seed)
    return model


def create_three_layer_net():
    config = None

    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(test_dir, 'data', 'test_config.yml')
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    loss_fn = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    model_class = models.ThreeLayerClassifier

    model = models.PytorchSupervised(config,
                                     optimizer,
                                     loss_fn,
                                     model_class)

    return model


@pytest.fixture(scope="module")
def dataset():
    return test_datasets.create_refinebio_labeled_dataset()


@pytest.fixture(scope="module")
def unlabeled_dataset():
    return test_datasets.create_refinebio_labeled_dataset()

def test_sklearn_fit_predict(sklearn_models, dataset):
    for model in sklearn_models:
        model = model.fit(dataset)

        unlabeled_dataset = datasets.RefineBioUnlabeledDataset.from_labeled_dataset(dataset)

        predictions = model.predict(unlabeled_dataset)
        assert type(predictions) == np.ndarray


def test_sklearn_save_load_model(sklearn_models, dataset):
    for model in sklearn_models:
        model.fit(dataset)

        unlabeled_dataset = datasets.RefineBioUnlabeledDataset.from_labeled_dataset(dataset)
        predictions = model.predict(unlabeled_dataset)

        model.save_model('model_test.pkl')
        loaded_model = type(model).load_model('model_test.pkl')

        new_predictions = loaded_model.predict(unlabeled_dataset)

        assert np.array_equal(predictions, new_predictions)

        os.remove('model_test.pkl')


def test_pytorch_save_load_model(pytorch_models, dataset, config):
    for model in pytorch_models:
        model.fit(dataset)

        unlabeled_dataset = datasets.RefineBioUnlabeledDataset.from_labeled_dataset(dataset)
        predictions = model.predict(unlabeled_dataset)

        model.save_model('model_test.pkl')
        # FIXME These are the same as in create_three_layer_net, but really the information should
        # probably be stored in the pickle file somehow
        optimizer = torch.optim.Adam
        loss_fn = torch.nn.CrossEntropyLoss
        model_class = models.ThreeLayerClassifier

        loaded_model = type(model).load_model('model_test.pkl',
                                              config,
                                              optimizer,
                                              loss_fn,
                                              model_class)
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


def test_pytorch_fit_predict(pytorch_models, dataset, config):
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


def test_unsupervised_fit_transform(unsupervised_models, dataset):
    for model in unsupervised_models:
        unlabeled_dataset = datasets.RefineBioUnlabeledDataset.from_labeled_dataset(dataset)
        X = unlabeled_dataset.get_all_data()

        embedded_data = model.fit_transform(unlabeled_dataset)

        embedded_X = embedded_data.get_all_data()

        # Ensure the number of samples is maintained
        assert X.shape[0] == embedded_X.shape[0]
        # Ensure dimension was reduced
        assert embedded_X.shape[1] == N_COMPONENTS

        # Ensure sample ids are preserved
        assert embedded_data.get_samples() == unlabeled_dataset.get_samples()


def test_pca_save_load(dataset):
    model = create_PCA()

    embedded_data = model.fit_transform(unlabeled_dataset)

