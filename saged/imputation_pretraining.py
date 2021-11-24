"""
This benchmark trains an imputation model and evaluates its performance in a binary disease
classification setting
"""
import argparse
import copy
import gc
import os

import sklearn.metrics
import yaml

from saged import utils, datasets, models


PREDICT_TISSUES = ['Blood', 'Breast', 'Stem Cell', 'Cervix', 'Brain', 'Kidney',
                   'Umbilical Cord', 'Lung', 'Epithelium', 'Prostate', 'Liver',
                   'Heart', 'Skin', 'Colon', 'Bone Marrow', 'Muscle', 'Tonsil',
                   'Blood Vessel', 'Spinal Cord', 'Testis', 'Placenta'
                   ]

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
    parser.add_argument('--num_splits',
                        help='The number of splits to use in cross-validation',
                        type=int,
                        default=5)
    parser.add_argument('--weighted_loss',
                        help='Weight classes based on the inverse of their prevalence',
                        action='store_true')

    args = parser.parse_args()

    with open(args.dataset_config) as data_file:
        dataset_config = yaml.safe_load(data_file)

    expression_df, sample_to_label, sample_to_study = utils.load_recount_data(args.dataset_config)
    all_data = datasets.RefineBioMixedDataset(expression_df, sample_to_label, sample_to_study)

    unlabeled_data = all_data.get_unlabeled()
    labeled_data = all_data.get_labeled()

    labeled_data.subset_samples_to_labels(PREDICT_TISSUES)

    labeled_data.recode()
    label_encoder = labeled_data.get_label_encoder()

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

        if args.weighted_loss:
            loss_weights = utils.calculate_loss_weights(labeled_data)

            # This won't affect the imputation model, but will kick in when to_classifier
            # is called (which is exactly what we want)
            model_config['loss_weights'] = loss_weights

    imputation_model_type = model_config.pop('name')
    SupervisedClass = getattr(models, imputation_model_type)
    imputation_model = SupervisedClass(**model_config)
    imputation_model.model = imputation_model.model.to('cpu')

    imputation_models.append((imputation_model, 0))
    subset_percents.append(0)

    # Imputation training loop
    # for subset_number in [1, 10]:
    for subset_number in [1]:
        subset_percent = subset_number * .1

        neptune_run = None
        # Parse config file
        if args.neptune_config is not None:
            with open(args.neptune_config) as neptune_file:
                neptune_config = yaml.safe_load(neptune_file)
                neptune_run = utils.initialize_neptune(neptune_config)

        train_data = unlabeled_data
        if subset_number < 10:
            train_data = train_data.subset_samples(subset_percent, args.seed)

        print('Train Samples: {}'.format(len(train_data.get_samples())))
        print('Train Studies: {}'.format(len(train_data.get_studies())))

        assert len(train_data) > 0

        input_size = train_data[0].shape[0]

        model_config['save_path'] += '/impute_{}_{}'.format(subset_percent, args.seed)
        imputation_model = SupervisedClass(**model_config)

        imputation_model.fit(train_data, neptune_run)

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

                train_data.set_label_encoder(label_encoder)
                val_data.set_label_encoder(label_encoder)

                # Now that the ratio is correct, actually subset the samples
                train_data = train_data.subset_samples(subset_percent,
                                                       args.seed)

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

                if imputation_model_copy.save_path is not None:
                    imputation_save = imputation_model_copy.save_path
                    imputation_save = imputation_save.split('impute')[0]

                    model_name = imputation_model.model_name
                    extra_information = '{}_{}_{}_{}'.format(model_name, i,
                                                             args.seed, impute_sample_count)
                    supervised_save = os.path.join(imputation_save, extra_information)
                    supervised_model.save_path = supervised_save

                supervised_model.fit(train_data, neptune_run)

                predictions, true_labels = supervised_model.evaluate(val_data)

                supervised_model.free_memory()
                imputation_model_copy.free_memory()

                accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
                balanced_acc = sklearn.metrics.balanced_accuracy_score(true_labels, predictions)

                accuracies.append(accuracy)
                balanced_accuracies.append(balanced_acc)
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
        out_file.write('accuracy\tbalanced_accuracy\ttrain studies\ttrain samples\t'
                       'val samples\ttrain sample count\tfraction of data used\timpute_samples\n')

        result_iterator = zip(accuracies,
                              balanced_accuracies,
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
