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
