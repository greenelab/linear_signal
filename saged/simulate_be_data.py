"""
Generate data for testing the effects of linear and nonlinear models on
data with the linear label signal removed
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_file', help='The location to store the simulated data')
    parser.add_argument('--debug', help='Pass this flag to visualize features',
                        action='store_true')
    parser.add_argument('--n_linear', help='The number of variables to include that have a linear '
                        'relationship with the label', default=2500, type=int)
    parser.add_argument('--n_nonlinear', help='The number of variables to include that have a '
                        'nonlinear relationship with the label', default=2500, type=int)

    args = parser.parse_args()

    N_SAMPLES = 1000
    N_LINEAR_VARIABLES = args.n_linear
    N_NONLINEAR_VARIABLES = args.n_nonlinear
    MEAN_DIFFERENCE = 6

    np.random.seed(42)

    labels = np.zeros(N_SAMPLES)
    labels[:N_SAMPLES // 2] = 1
    labels = ['red' if lab == 0 else 'blue' for lab in labels]

    vars = {}

    # Linear variables
    for i in range(N_LINEAR_VARIABLES):
        current_var = []
        for j in range(N_SAMPLES):
            if labels[j] == 'red':
                current_var.append(np.random.normal(MEAN_DIFFERENCE))
            else:
                current_var.append(np.random.normal(0))

        vars['linear_{}'.format(i)] = current_var

    for i in range(N_NONLINEAR_VARIABLES):
        current_var = []
        for j in range(N_SAMPLES):
            if labels[j] == 'red':
                if np.random.random() > .5:
                    current_var.append(np.random.normal(MEAN_DIFFERENCE))
                else:
                    current_var.append(np.random.normal(-MEAN_DIFFERENCE))
            else:
                current_var.append(np.random.normal(0))
        vars['nonlinear_{}'.format(i)] = current_var

    rownames = ['sample_{}'.format(i) for i in range(N_SAMPLES)]
    data = pd.DataFrame(vars, index=rownames)
    data['label'] = labels

    if args.debug:
        hist = data['linear_94'].hist(bins=40)
        plt.show()
        hist = data['nonlinear_93'].hist(bins=40)
        plt.show()

    data.to_csv(args.out_file, sep='\t')
