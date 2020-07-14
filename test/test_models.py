""" Test the functions implemented in models.py """

import os

import numpy as np
import pytest

import test_datasets
from saged import models, datasets


@pytest.fixture(scope="module")
def sklearn_models():
    model_list = []
    model_list.append(models.LogisticRegression())

    return model_list


@pytest.fixture(scope="module")
def pytorch_models():
    model_list = []
    # Pytorch models go here

    return model_list


@pytest.fixture(scope="module")
def all_models(sklearn_models, pytorch_models):
    return sklearn_models + pytorch_models


@pytest.fixture(scope="module")
def dataset():
    return test_datasets.create_refinebio_labeled_dataset()


def test_sklearn_fit_predict(sklearn_models, dataset):
    for model in sklearn_models:
        model = model.fit(dataset)

        unlabeled_dataset = datasets.RefineBioUnlabeledDataset.from_labeled_dataset(dataset)

        predictions = model.predict(unlabeled_dataset)
        assert type(predictions) == np.ndarray


def test_save_load_model(all_models, dataset):
    for model in all_models:
        model.fit(dataset)

        unlabeled_dataset = datasets.RefineBioUnlabeledDataset.from_labeled_dataset(dataset)
        predictions = model.predict(unlabeled_dataset)

        model.save_model('model_test.pkl')
        loaded_model = type(model).load_model('model_test.pkl')

        new_predictions = loaded_model.predict(unlabeled_dataset)

        assert np.array_equal(predictions, new_predictions)

        os.remove('model_test.pkl')


def test_load_params(pytorch_models):
    raise NotImplementedError


def test_pytorch_fit_predict(pytorch_models, dataset):
    raise NotImplementedError
