"""
This benchmark compares the performance of different models in learning to differentiate between
healthy and diseased gene expression with equal label distributions in the train and val sets."""
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
    parser.add_argument('--semi_supervised',
                        help='This flag tells the script that the config file passed in is a '
                             'semi-supervised model',
                        action='store_true',
                        default=False)
    parser.add_argument('--batch_correction_method',
                        help='The method to use to correct for batch effects',
                        default=None)

    args = parser.parse_args()

    with open(args.dataset_config) as data_file:
        dataset_config = yaml.safe_load(data_file)

    # Parse config file
    if args.neptune_config is not None:
        with open(args.neptune_config) as neptune_file:
            neptune_config = yaml.safe_load(neptune_file)
            utils.initialize_neptune(neptune_config)

    all_data, labeled_data, unlabeled_data = datasets.load_binary_data(args.dataset_config,
                                                                       args.label,
                                                                       args.negative_class)
    # Load binary data subsets the data to two classes. Update the label encoder to treat this
    # data as binary so the F1 score doesn't break
    labeled_data.recode()
    label_encoder = labeled_data.get_label_encoder()

    # Correct for batch effects
    if args.batch_correction_method is not None:
        all_data = datasets.correct_batch_effects(all_data, args.batch_correction_method)
        labeled_data = all_data.get_labeled()
        labeled_data.subset_samples_to_labels([args.label, args.negative_class])
        unlabeled_data = all_data.get_unlabeled()

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
        for samples_to_keep_count in range(10, 200, 10):
            train_list = labeled_splits[:i] + labeled_splits[i+1:]

            # Extract the train and test datasets
            LabeledDatasetClass = type(labeled_data)
            train_data = LabeledDatasetClass.from_list(train_list)
            val_data = labeled_splits[i]

            # This isn't strictly necessary since we're checking whether both classes are present,
            # but it's safer
            train_data.set_label_encoder(label_encoder)
            val_data.set_label_encoder(label_encoder)

            train_data.keep_k_samples_for_label(samples_to_keep_count,
                                                args.label,
                                                args.seed)
            train_data.keep_k_samples_for_label(samples_to_keep_count,
                                                args.negative_class,
                                                args.seed)

            # Only use the training data if there were enough samples in both classes
            if len(train_data) != 2 * samples_to_keep_count:
                continue

            # Skip entries where there is only data for one class
            if len(train_data.get_classes()) <= 1 or len(val_data.get_classes()) <= 1:
                continue

            val_data = utils.subset_to_equal_ratio(val_data, train_data, args.label,
                                                   args.negative_class, args.seed)

            print('Samples: {}'.format(len(train_data.get_samples())))
            print('Studies: {}'.format(len(train_data.get_studies())))
            if args.semi_supervised:
                print('Unlabeled samples: {}'.format(len(unlabeled_data.get_samples())))

            print('Val data: {}'.format(len(val_data)))
            input_size = len(train_data.get_features())
            output_size = len(train_data.get_classes())
            print('output size: {}'.format(output_size))

            if args.unsupervised_config is not None:
                with open(args.unsupervised_config) as unsupervised_file:
                    unsupervised_config = yaml.safe_load(unsupervised_file)

                train_data, val_data, unsupervised_model = models.embed_data(unsupervised_config,
                                                                             all_data,
                                                                             train_data,
                                                                             unlabeled_data,
                                                                             val_data)

                # Adjust the input size since we aren't using the original dimensionality
                input_size = len(train_data.get_features())

            with open(args.supervised_config) as supervised_file:
                supervised_config = yaml.safe_load(supervised_file)
                supervised_config['input_size'] = input_size
                supervised_config['output_size'] = output_size

            supervised_model_type = supervised_config.pop('name')
            SupervisedClass = getattr(models, supervised_model_type)
            supervised_model = SupervisedClass(**supervised_config)

            # If the model is semi-supervised, train it with the train data and the unlabeled data
            if args.semi_supervised:
                all_data = all_data.subset_to_samples(train_data.get_samples() +
                                                      unlabeled_data.get_samples())
                supervised_model.fit(all_data)
                all_data.reset_filters()
            # Train the model on the training data
            else:
                supervised_model.fit(train_data)

            predictions, true_labels = supervised_model.evaluate(val_data)

            supervised_model.free_memory()

            accuracy = sklearn.metrics.accuracy_score(predictions, true_labels)
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

            train_data.reset_filters()
            val_data.reset_filters()

    with open(args.out_file, 'w') as out_file:
        # Write header
        out_file.write('accuracy\tbalanced_accuracy\tf1_score\ttrain studies\ttrain samples\t')
        out_file.write('val samples\ttrain sample count\n')

        result_iterator = zip(accuracies,
                              balanced_accuracies,
                              f1_scores,
                              supervised_train_studies,
                              supervised_train_sample_names,
                              supervised_val_sample_names,
                              supervised_train_sample_counts,
                              )
        for stats in result_iterator:
            stat_strings = [str(item) for item in stats]
            out_str = '\t'.join(stat_strings)
            out_file.write(f'{out_str}\n')
