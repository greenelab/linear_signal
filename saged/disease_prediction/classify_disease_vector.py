"""
Run classifiers on the output of disease_vector.py
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import sklearn.metrics
import yaml

from saged import utils, datasets, models

# Load dataset
# Create dataframes from datasets
# Create sample_to_label and sample_to_study
# Generate dataset objects

# For dataset in datasets:
#   Create CV splits
#   Create model
#
#   train model on dataset splits
#   Log results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('simulated_healthy_file',
                        help='A pkl file containing a genes x samples matrix of simulated '
                             'healthy expression data')
    parser.add_argument('simulated_disease_file',
                        help='A pkl file containing a genes x samples matrix of simulated '
                             'disease expression data')
    parser.add_argument('supervised_config',
                        help='The yaml formatted model configuration file. For more information '
                             'about this file read the comments in the example_model.yml file')
    parser.add_argument('out_file',
                        help='The file to save the results to')
    parser.add_argument('--label',
                        help='The label to be predicted from the data',
                        default='sepsis')
    parser.add_argument('--negative_class',
                        help='The baseline class to predict the label against. For example '
                             'in the refinebio dataset the negative class is "healthy"',
                        default='healthy')
    parser.add_argument('--neptune_config',
                        help='A yaml formatted file containing init information for '
                             'neptune logging')
    parser.add_argument('--seed',
                        help='The random seed to be used in splitting data',
                        type=int,
                        default=42)
    parser.add_argument('--num_splits',
                        help='The number of splits to use in cross-validation',
                        type=int,
                        default=5)
    parser.add_argument('--semi_supervised',
                        help='This flag tells the script that the config file passed in is a '
                             'semi-supervised model',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    with open(args.simulated_healthy_file, 'rb') as healthy_file:
        healthy_arr = pickle.load(healthy_file)

    with open(args.simulated_disease_file, 'rb') as disease_file:
        disease_arr = pickle.load(disease_file)

    healthy_ids = ['{}_{}'.format(args.negative_class, i) for i in range(healthy_arr.shape[1])]
    disease_ids = ['{}_{}'.format(args.label, i) for i in range(disease_arr.shape[1])]

    sample_to_label = {}
    sample_to_study = {}
    for i, id_ in enumerate(healthy_ids):
        sample_to_label[id_] = args.negative_class
        study = '{}_STUDY_{}'.format(args.negative_class, i)
        sample_to_study[id_] = study
    for i, id_ in enumerate(disease_ids):
        sample_to_label[id_] = args.label
        study = '{}_STUDY_{}'.format(args.label, i)
        sample_to_study[id_] = study

    all_data_arr = np.concatenate((healthy_arr, disease_arr), axis=1)

    data_df = pd.DataFrame(all_data_arr, columns=healthy_ids + disease_ids)
    labeled_data = datasets.RefineBioLabeledDataset(data_df,
                                                    sample_to_label=sample_to_label,
                                                    sample_to_study=sample_to_study)

    # Get fivefold cross-validation splits
    labeled_splits = labeled_data.get_cv_splits(num_splits=args.num_splits, seed=args.seed)

    # Train the model on each fold
    accuracies = []
    balanced_accuracies = []
    f1_scores = []
    supervised_train_studies = []
    supervised_train_sample_names = []
    supervised_val_sample_names = []
    supervised_train_sample_counts = []
    subset_percents = []
    for i in range(len(labeled_splits)):
        for subset_number in range(1, 11, 2):
            # The new neptune version doesn't have a create_experiment function so you have to
            # reinitialize per-model
            neptune_run = None
            # Parse config file
            if args.neptune_config is not None:
                with open(args.neptune_config) as neptune_file:
                    neptune_config = yaml.safe_load(neptune_file)
                    neptune_run = utils.initialize_neptune(neptune_config)

            subset_percent = subset_number * .1

            train_list = labeled_splits[:i] + labeled_splits[i+1:]

            # Extract the train and test datasets
            LabeledDatasetClass = type(labeled_data)
            train_data = LabeledDatasetClass.from_list(train_list)
            val_data = labeled_splits[i]

            train_data = utils.subset_to_equal_ratio(train_data, val_data, args.label,
                                                     args.negative_class, args.seed)
            # Now that the ratio is correct, actually subset the samples
            train_data = train_data.subset_samples(subset_percent,
                                                   args.seed)

            # Skip entries where there is only data for one class
            if len(train_data.get_classes()) <= 1 or len(val_data.get_classes()) <= 1:
                continue

            print('Samples: {}'.format(len(train_data.get_samples())))
            print('Studies: {}'.format(len(train_data.get_studies())))

            print('Val data: {}'.format(len(val_data)))
            input_size = len(train_data.get_features())
            output_size = len(train_data.get_classes())
            print('output size: {}'.format(output_size))

            with open(args.supervised_config) as supervised_file:
                supervised_config = yaml.safe_load(supervised_file)
                supervised_config['input_size'] = input_size
                supervised_config['output_size'] = output_size

            supervised_model_type = supervised_config.pop('name')
            SupervisedClass = getattr(models, supervised_model_type)
            supervised_model = SupervisedClass(**supervised_config)

            supervised_model.fit(train_data, neptune_run)

            predictions, true_labels = supervised_model.evaluate(val_data)

            supervised_model.free_memory()

            accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
            positive_label_encoding = train_data.get_label_encoding(args.label)
            balanced_acc = sklearn.metrics.balanced_accuracy_score(true_labels, predictions)
            f1_score = sklearn.metrics.f1_score(true_labels, predictions,
                                                pos_label=positive_label_encoding,
                                                average='binary')

            accuracies.append(accuracy)
            balanced_accuracies.append(balanced_acc)
            f1_scores.append(f1_score)
            supervised_train_studies.append(','.join(train_data.get_studies()))
            supervised_train_sample_names.append(','.join(train_data.get_samples()))
            supervised_val_sample_names.append(','.join(val_data.get_samples()))
            supervised_train_sample_counts.append(len(train_data))
            subset_percents.append(subset_percent)

            train_data.reset_filters()
            val_data.reset_filters()

    with open(args.out_file, 'w') as out_file:
        # Write header
        out_file.write('accuracy\tbalanced_accuracy\tf1_score\ttrain studies\ttrain samples\t'
                       'val samples\ttrain sample count\tfraction of data used\n')

        result_iterator = zip(accuracies,
                              balanced_accuracies,
                              f1_scores,
                              supervised_train_studies,
                              supervised_train_sample_names,
                              supervised_val_sample_names,
                              supervised_train_sample_counts,
                              subset_percents
                              )
        for stats in result_iterator:
            stat_strings = [str(item) for item in stats]
            out_str = '\t'.join(stat_strings)
            out_file.write(f'{out_str}\n')
