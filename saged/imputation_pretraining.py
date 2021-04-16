"""
This benchmark trains an imputation model for downstream use in classification
"""
import argparse
import copy
import gc

import sklearn.metrics
import yaml

from saged import utils, datasets, models


def subset_to_equal_ratio(train_data: datasets.LabeledDataset,
                          val_data: datasets.LabeledDataset
                          ) -> datasets.LabeledDataset:
    """
    Subset the training dataset to match the ratio of positive to negative expression samples in
    the validation dataset

    Arguments
    ---------
    train_data: The train expression dataset
    val_data: The validation expression dataset

    Returns
    -------
    train_data: The subsetted expression dataset
    """

    train_disease_counts = train_data.map_labels_to_counts()
    val_disease_counts = val_data.map_labels_to_counts()

    train_positive = train_disease_counts.get(args.label, 0)
    train_negative = train_disease_counts.get(args.negative_class, 0)
    val_positive = val_disease_counts.get(args.label, 0)
    val_negative = val_disease_counts.get(args.negative_class, 0)

    train_disease_fraction = train_positive / (train_positive + train_negative)
    val_disease_fraction = val_positive / (val_positive + val_negative)

    subset_fraction = utils.determine_subset_fraction(train_positive,
                                                      train_negative,
                                                      val_positive,
                                                      val_negative)

    # If train ratio is too high, remove positive samples
    if train_disease_fraction > val_disease_fraction:
        train_data = train_data.subset_samples_for_label(subset_fraction,
                                                         args.label,
                                                         args.seed)
    # If train ratio is too low, remove negative samples
    elif train_disease_fraction < val_disease_fraction:
        train_data = train_data.subset_samples_for_label(subset_fraction,
                                                         args.negative_class,
                                                         args.seed)
    return train_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_config',
                        help='The yaml formatted dataset configuration file. For more information '
                             'about this file read the comments in the example_dataset.yml file')
    parser.add_argument('model_config',
                        help='The yaml formatted model configuration file. For more information '
                             'about this file read the comments in the example_model.yml file')
    parser.add_argument('out_file',
                        help='The file to save the results to')
    parser.add_argument('--neptune_config',
                        help='A yaml formatted file containing init information for '
                             'neptune logging')
    parser.add_argument('--seed',
                        help='The random seed to be used in splitting data',
                        type=int,
                        default=42)
    parser.add_argument('--label',
                        help='The label to be predicted from the data',
                        default='sepsis')
    parser.add_argument('--negative_class',
                        help='The baseline class to predict the label against. For example '
                             'in the refinebio dataset the negative class is "healthy"',
                        default='healthy')
    parser.add_argument('--num_splits',
                        help='The number of splits to use in cross-validation',
                        type=int,
                        default=5)
    parser.add_argument('--batch_correction_method',
                        help='The method to use to correct for batch effects',
                        default=None)

    args = parser.parse_args()

    # TODO
    # Write logic to convert imputation model to classification model
    # ^(should live in PytorchSupervised probably)
    # Write logic to hold out samples in imputation
    # Create new script to run trained model for classification?

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
    # Correct for batch effects
    if args.batch_correction_method is not None:
        all_data = datasets.correct_batch_effects(all_data, args.batch_correction_method)

    labeled_samples = labeled_data.get_samples()
    all_data = all_data.remove_samples(labeled_samples)

    labeled_data.recode()
    label_encoder = labeled_data.get_label_encoder()

    # Train imputation model
    # Save imputation results
    # Create classification model from imputation model
    # Train classification model
    #

    # Train the model on each fold
    train_studies = []
    train_sample_names = []
    val_sample_names = []
    train_sample_counts = []
    subset_percents = []
    val_losses = []

    imputation_models = []

    # Add an untrained model to the list
    subset_percent = 0
    input_size = all_data[0].shape[0]
    with open(args.model_config) as supervised_file:
        model_config = yaml.safe_load(supervised_file)
        model_config['input_size'] = input_size
        # Output size is the same as the input because we're doing
        # imputation
        model_config['output_size'] = input_size
        model_config['save_path'] += '/impute_{}_{}'.format(subset_percent, args.seed)

    imputation_model_type = model_config.pop('name')
    SupervisedClass = getattr(models, imputation_model_type)
    imputation_model = SupervisedClass(**model_config)
    imputation_model.model = imputation_model.model.to('cpu')

    imputation_models.append((imputation_model, 0))
    subset_percents.append(0)

    # Imputation training loop
    for subset_number in [1, 10]:
        subset_percent = subset_number * .1

        # TODO fix this logic if you want to track impute performance
        # val_data isn't necessary for model training b/c a tune set is pulled out in the
        # train function

        # train_list = cv_splits[:i] + cv_splits[i+1:]

        # # Extract the train and test datasets
        # DatasetClass = type(all_data)
        # train_data = DatasetClass.from_list(train_list)
        # val_data = cv_splits[i]

        train_data = all_data
        train_data = train_data.subset_samples(subset_percent, args.seed)

        print('Train Samples: {}'.format(len(train_data.get_samples())))
        print('Train Studies: {}'.format(len(train_data.get_studies())))

        assert len(all_data) > 0

        input_size = all_data[0].shape[0]

        imputation_model = SupervisedClass(**model_config)

        imputation_model.fit(train_data)

        # Keep model from taking up GPU space
        imputation_model.model = imputation_model.model.to('cpu')
        imputation_models.append((imputation_model, len(train_data)))

        train_studies.append(','.join(train_data.get_studies()))
        train_sample_names.append(','.join(train_data.get_samples()))
        train_sample_counts.append(len(train_data))
        subset_percents.append(subset_percent)

        train_data.reset_filters()

    # all_data is a lot of memory to hang on to, so don't
    del(all_data)
    del(unlabeled_data)

    # Get fivefold cross-validation splits
    labeled_splits = labeled_data.get_cv_splits(num_splits=args.num_splits, seed=args.seed)

    # Classification training loop
    accuracies = []
    balanced_accuracies = []
    f1_scores = []
    supervised_train_studies = []
    supervised_train_sample_names = []
    supervised_val_sample_names = []
    supervised_train_sample_counts = []
    subset_percents = []
    impute_sample_counts = []

    for i in range(len(labeled_splits)):
        for subset_number in range(1, 11, 1):
            for imputation_model, impute_sample_count in imputation_models:
                # https://github.com/facebookresearch/higher/pull/15
                gc.collect()

                subset_percent = subset_number * .1

                train_list = labeled_splits[:i] + labeled_splits[i+1:]

                # Extract the train and test datasets
                LabeledDatasetClass = type(labeled_data)
                train_data = LabeledDatasetClass.from_list(train_list)
                val_data = labeled_splits[i]

                # This isn't strictly necessary since we're checking whether both classes
                # are present, but it's safer
                train_data.set_label_encoder(label_encoder)
                val_data.set_label_encoder(label_encoder)

                train_data = subset_to_equal_ratio(train_data, val_data)
                # Now that the ratio is correct, actually subset the samples
                train_data = train_data.subset_samples(subset_percent,
                                                       args.seed)

                # Skip entries where there is only data for one class
                if len(train_data.get_classes()) <= 1 or len(val_data.get_classes()) <= 1:
                    continue

                print('Samples: {}'.format(len(train_data.get_samples())))
                print('Studies: {}'.format(len(train_data.get_studies())))

                output_size = len(train_data.get_classes())

                print('Val data: {}'.format(len(val_data)))
                print('output size: {}'.format(output_size))

                # Copy the model before converting it to be a classifier to prevent
                # retraining the same model repeatedly
                imputation_model_copy = copy.deepcopy(imputation_model)
                supervised_model = imputation_model_copy.to_classifier(output_size,
                                                                       'CrossEntropyLoss')

                supervised_model.fit(train_data)

                predictions, true_labels = supervised_model.evaluate(val_data)

                supervised_model.free_memory()
                imputation_model_copy.free_memory()

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
                impute_sample_counts.append(impute_sample_count)

                train_data.reset_filters()
                val_data.reset_filters()


    with open(args.out_file, 'w') as out_file:
        # Write header
        out_file.write('accuracy\tbalanced_accuracy\tf1_score\ttrain studies\ttrain samples\t'
                       'val samples\ttrain sample count\tfraction of data used\timpute_samples\n')

        result_iterator = zip(accuracies,
                              balanced_accuracies,
                              f1_scores,
                              supervised_train_studies,
                              supervised_train_sample_names,
                              supervised_val_sample_names,
                              supervised_train_sample_counts,
                              subset_percents,
                              impute_sample_counts
                              )
        for stats in result_iterator:
            stat_strings = [str(item) for item in stats]
            out_str = '\t'.join(stat_strings)
            out_file.write(f'{out_str}\n')
