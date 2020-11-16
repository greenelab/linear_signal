""" Test utility functions """

import numpy as np
import pytest
import torch

from saged import utils


@pytest.mark.parametrize('labels, true_correct',
                         [
                          ([1, 0, 3, 3], 1),
                          ([0, 0, 0, 0], .25),
                          ([2, 3, 1, 2], 0),
                         ])
def test_count_correct(labels, true_correct):
    outputs = np.array([[.1, .5, .3, 0],
                        [.9, .2, .2, .2],
                        [0, 0, 0, 0],
                        [1, 2, 3, 4]
                        ])
    outputs = torch.Tensor(outputs)
    labels = torch.Tensor(labels)

    correct = utils.count_correct(outputs, labels)
    assert correct == true_correct


@pytest.mark.parametrize('train_positive, train_negative, val_positive, val_negative',
                         [
                          (5, 200, 1, 9),
                          (100, 200, 1, 9),
                          (600, 200, 1, 9),
                          (1000, 10, 1, 9),
                          (5, 200, 9, 1),
                          (100, 200, 9, 1),
                          (600, 200, 9, 1),
                          (1000, 10, 9, 1),
                         ])
def determine_subset_fraction(train_positive, train_negative, val_positive, val_negative):
    subset_fraction = utils.determine_subset_fraction(train_positive, train_negative,
                                                      val_positive, val_negative)

    assert subset_fraction <= 1

    train_fraction = train_positive / (train_positive + train_negative)
    val_fraction = val_positive / (val_positive + val_negative)

    if train_fraction > val_fraction:
        subset_result = subset_fraction * train_positive
        new_train_frac = subset_result / (subset_result + train_negative)
    else:
        subset_result = subset_fraction * train_negative
        new_train_frac = train_positive / (subset_result + train_positive)

    assert pytest.approx(new_train_frac, val_fraction)
