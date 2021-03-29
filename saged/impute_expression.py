"""
This benchmark compares the performance of different models in learning to differentiate between
healthy and diseased gene expression with equal label distributions in the train and val sets."""
import argparse

import yaml

from saged import utils, datasets, models


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
    parser.add_argument('--batch_correction_method',
                        help='The method to use to correct for batch effects',
                        default=None)

    args = parser.parse_args()

    # TODO
    # Evaluate effects of study splits (am I correct to split by study to avoid
    # leakage?)
    # Evaluate effects of amount of training data
    # Implement adding noise?

    with open(args.dataset_config) as data_file:
        dataset_config = yaml.safe_load(data_file)

    # Parse config file
    if args.neptune_config is not None:
        with open(args.neptune_config) as neptune_file:
            neptune_config = yaml.safe_load(neptune_file)
            utils.initialize_neptune(neptune_config)

    all_data = datasets.load_all_data(args.dataset_config)

    # Correct for batch effects
    if args.batch_correction_method is not None:
        all_data = datasets.correct_batch_effects(all_data, args.batch_correction_method)

    # Get fivefold cross-validation splits
    cv_splits = all_data.get_cv_splits(num_splits=args.num_splits, seed=args.seed)

    # Train the model on each fold
    train_studies = []
    train_sample_names = []
    val_sample_names = []
    train_sample_counts = []
    subset_percents = []
    val_losses = []
    for i in range(len(cv_splits)):
        for subset_number in range(1, 11):
            subset_percent = subset_number * .1

            train_list = cv_splits[:i] + cv_splits[i+1:]

            # Extract the train and test datasets
            DatasetClass = type(all_data)
            train_data = DatasetClass.from_list(train_list)
            val_data = cv_splits[i]

            train_data = train_data.subset_samples(subset_percent, args.seed)

            print('Train Samples: {}'.format(len(train_data.get_samples())))
            print('Train Studies: {}'.format(len(train_data.get_studies())))

            assert len(all_data) > 0

            input_size = all_data[0].shape[0]

            with open(args.model_config) as supervised_file:
                model_config = yaml.safe_load(supervised_file)
                model_config['input_size'] = input_size
                # Output size is the same as the input because we're doing
                # imputation
                model_config['output_size'] = input_size

            supervised_model_type = model_config.pop('name')
            SupervisedClass = getattr(models, supervised_model_type)
            supervised_model = SupervisedClass(**model_config)

            supervised_model.fit(train_data)

            val_loss = supervised_model.evaluate(val_data)

            supervised_model.free_memory()

            train_studies.append(','.join(train_data.get_studies()))
            train_sample_names.append(','.join(train_data.get_samples()))
            val_sample_names.append(','.join(val_data.get_samples()))
            train_sample_counts.append(len(train_data))
            subset_percents.append(subset_percent)
            val_losses.append(val_loss)

            train_data.reset_filters()
            val_data.reset_filters()

    with open(args.out_file, 'w') as out_file:
        # Write header
        out_file.write('val_loss\ttrain studies\ttrain samples\t'
                       'val samples\ttrain sample count\tfraction of data used\n')

        result_iterator = zip(val_losses,
                              train_studies,
                              train_sample_names,
                              val_sample_names,
                              train_sample_counts,
                              subset_percents
                              )
        for stats in result_iterator:
            stat_strings = [str(item) for item in stats]
            out_str = '\t'.join(stat_strings)
            out_file.write(f'{out_str}\n')
