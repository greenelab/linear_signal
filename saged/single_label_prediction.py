"""
This benchmark compares the performance of different models in learning to predict all the
classes in the dataset
"""
import argparse

import sklearn.metrics
import yaml

from saged import utils, datasets, models


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
    parser.add_argument('--label',
                        help='The label to be predicted from the data',
                        default='sepsis')
    parser.add_argument('--negative_class',
                        help='The baseline class to predict the label against. For example '
                             'in the refinebio dataset the negative class is "healthy"',
                        default='healthy')
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
                        help='The random seed to be used in splitting data',
                        type=int,
                        default=42)
    parser.add_argument('--num_splits',
                        help='The number of splits to use in cross-validation',
                        type=int,
                        default=5)
    args = parser.parse_args()

    with open(args.dataset_config) as data_file:
        dataset_config = yaml.safe_load(data_file)

    with open(args.supervised_config) as supervised_file:
        supervised_config = yaml.safe_load(supervised_file)

    # Parse config file

    if args.neptune_config is not None:
        with open(args.neptune_config) as neptune_file:
            neptune_config = yaml.safe_load(neptune_file)
            utils.initialize_neptune(neptune_config)

    # Get the class of dataset to use with this configuration
    dataset_name = dataset_config.pop('name')
    MixedDatasetClass = getattr(datasets, dataset_name)

    print('Loading all data')
    all_data = MixedDatasetClass.from_config(**dataset_config)
    print('Subsetting labeled data')
    labeled_data = all_data.get_labeled()
    labeled_data.subset_samples_to_labels([args.label, args.negative_class])
    print('Subsetting unlabeled data')
    unlabeled_data = all_data.get_unlabeled()
    print('Splitting data')

    # Get fivefold cross-validation splits
    labeled_splits = labeled_data.get_cv_splits(num_splits=args.num_splits, seed=args.seed)

    # Train the model on each fold
    accuracies = []
    supervised_train_studies = []
    supervised_train_sample_counts = []
    for i in range(len(labeled_splits)):
        train_list = labeled_splits[:i] + labeled_splits[i+1:]

        # Extract the train and test datasets
        LabeledDatasetClass = type(labeled_data)
        train_data = LabeledDatasetClass.from_list(train_list)
        print('Samples: {}'.format(len(train_data.get_samples())))
        print('Studies: {}'.format(len(train_data.get_studies())))
        val_data = labeled_splits[i]

        input_size = len(train_data.get_features())
        output_size = len(train_data.get_classes())
        print('output size: {}'.format(output_size))

        if args.unsupervised_config is not None:
            with open(args.unsupervised_config) as unsupervised_file:
                unsupervised_config = yaml.safe_load(unsupervised_file)

            # Initialize the unsupervised model
            unsupervised_model_type = unsupervised_config.pop('name')
            UnsupervisedClass = getattr(models, unsupervised_model_type)
            unsupervised_model = UnsupervisedClass(**unsupervised_config)

            # Get all data not held in the val split
            available_data = all_data.subset_to_samples(train_data.get_samples() +
                                                        unlabeled_data.get_samples())

            # Embed the training data
            unsupervised_model.fit(available_data)
            train_data = unsupervised_model.transform(train_data)

            # Embed the validation data
            val_data = unsupervised_model.transform(val_data)

            # Adjust the input size since we aren't using the original dimensionality
            input_size = len(train_data.get_features())

            # Reset filters on all_data which were changed to create available_data
            all_data.reset_filters()

        with open(args.supervised_config) as supervised_file:
            supervised_config = yaml.safe_load(supervised_file)
            supervised_config['input_size'] = input_size
            supervised_config['output_size'] = output_size

        supervised_model_type = supervised_config.pop('name')
        SupervisedClass = getattr(models, supervised_model_type)
        supervised_model = SupervisedClass(**supervised_config)

        # Train the model on the training data
        supervised_model.fit(train_data)
        predictions, true_labels = supervised_model.evaluate(val_data)

        supervised_model.free_memory()

        accuracy = sklearn.metrics.accuracy_score(predictions, true_labels)

        accuracies.append(accuracy)
        supervised_train_studies.append(','.join(train_data.get_studies()))
        supervised_train_sample_counts.append(len(train_data))

    with open(args.out_file, 'w') as out_file:
        out_file.write('accuracy\ttrain studies\ttrain sample count\n')
        for (accuracy,
             train_study_str,
             supervised_train_samples) in zip(accuracies,
                                              supervised_train_studies,
                                              supervised_train_sample_counts):
            out_file.write(f'{accuracy}\t{train_study_str}\t{supervised_train_samples}\n')
