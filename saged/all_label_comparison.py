"""
This benchmark compares the performance of different models in learning to predict all the
classes in the dataset
"""
import argparse

import sklearn.metrics
import yaml

from saged import utils, datasets, models

# TODO make a mixed data class for storing labeled and unlabeled data
# TODO make sure mixed data class has a subset_to_samples method
# TODO have mixed data class return labels with -1 for unlabeled
# TODO change PCA class to adapt to accept mixed data
# TODO update tests

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_config',
                        help='The yaml formatted dataset configuration file. For more information '
                             'about this file read the comments in the example_dataset.yml file')
    parser.add_argument('supervised_config',
                        help='The yaml formatted model configuration file. For more information '
                             'about this file read the comments in the example_model.yml file')
    parser.add_argument('out_file',
                        help='The file to save the results to')
    parser.add_argument('--unsupervised_config',
                        help='The yaml formatted model configuration file for an unsupervised '
                             'model. If omitted, the benchmark will use the original features '
                             'for training. For more information about this file read the '
                             'comments in the example_model.yml file',
                        default=None)
    parser.add_argument('--neptune_config',
                        help='A yaml formatted file containing init information for '
                             'neptune logging')
    parser.add_argument('--seed',
                        help='The random seed to be used in the experiment',
                        type=int,
                        default=42)
    args = parser.parse_args()

    dataset_config = yaml.safe_load(args.dataset_config)
    supervised_config = yaml.safe_load(args.supervised_config)

    # Parse config file
    # Load dataset

    neptune_config_file = getattr('neptune_config', args)
    if neptune_config_file is not None:
        neptune_config = yaml.safe_load(neptune_config_file)
        utils.initialize_neptune(neptune_config)

    # Get the class of dataset to use with this configuration
    DatasetClass = getattr(dataset_config['name'], datasets)
    all_data = DatasetClass(dataset_config)
    labeled_data = all_data.get_labeled()
    unlabeled_data = all_data.get_unlabeled()

    # Get fivefold cross-validation splits
    labeled_splits = labeled_data.get_cv_splits(num_splits=5, seed=args.seed)

    # Train the model on each fold
    accuracies = []
    train_studies = []
    train_sample_counts = []
    for i in range(len(labeled_splits)):
        train_list = labeled_splits[:i] + labeled_splits[i+1:]

        # Extract the train and test datasets
        train_data = DatasetClass.from_datasets(train_list)
        val_data = labeled_splits[i]

        if args.unsupervised_config is not None:
            unsupervised_config = yaml.safe_load(args.unsupervised_config)

            # Initialize the unsupervised model
            UnsupervisedClass = getattr(unsupervised_config['name'], models)
            unsupervised_model = UnsupervisedClass(unsupervised_config)

            # Get all data not held in the val split
            available_data = all_data.subset_to_samples(train_data.get_samples() +
                                                        unlabeled_data.get_samples())

            # Embed the training data
            unsupervised_model.fit(available_data)
            train_data = unsupervised_model.transform(train_data)

        SupervisedClass = getattr(supervised_config['name'], models)
        supervised_model = SupervisedClass(supervised_config)

        # Train the model on the training data
        supervised_model.fit(train_data)

        predictions, true_labels = supervised_model.evaluate(val_data)

        # TODO more measures than Top-1 accuracy
        accuracy = sklearn.metrics.accuracy_score(predictions, true_labels)

        accuracies.append(accuracy)
        train_studies.append(','.join(train_data.get_studies()))
        train_sample_counts.append(len(train_data))

    with open(args.out_file, 'w') as out_file:
        out_file.write('accuracy\ttrain studies\ttrain sample count\n')
        for accuracy, train_study_str, train_samples in zip(accuracies,
                                                            train_studies,
                                                            train_sample_counts):
            out_file.write(f'{accuracy}\t{train_study_str}\t{train_samples}\n')
