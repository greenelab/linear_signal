"""
This benchmark trains two models, one trained on half the labeled data then retrained on the other
half, and one only trained on the second half. This acts as a positive control showing that
pretraining has a positive effect on model accuracy under ideal conditions
"""
import argparse
import copy
import gc

import sklearn.metrics
import torch

import yaml

from saged import utils, datasets, models
from saged.models import LogisticRegression


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

    args = parser.parse_args()

    with open(args.dataset_config) as data_file:
        dataset_config = yaml.safe_load(data_file)

    expression_df, sample_to_label, sample_to_study = utils.load_recount_data(args.dataset_config)
    all_data = datasets.RefineBioMixedDataset(expression_df, sample_to_label, sample_to_study)

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

    input_size = all_data[0].shape[0]
    with open(args.model_config) as supervised_file:
        model_config = yaml.safe_load(supervised_file)
        model_config['input_size'] = input_size
        # Output size is the same as the input because we're doing
        # imputation
        model_config['output_size'] = input_size

    # Split the data into two sets
    labeled_splits = labeled_data.get_cv_splits(num_splits=5,
                                                seed=args.seed,
                                                split_by_sample=True)

    del(all_data)

    accuracies = []
    balanced_accuracies = []
    pretrain_sample_names = []
    train_studies = []
    train_sample_names = []
    val_sample_names = []
    train_sample_counts = []
    subset_percents = []
    pretrained_or_trained = []

    for i in range(len(labeled_splits)):
        for subset_number in range(1, 11, 1):
            imputation_model_type = model_config['name']
            SupervisedClass = getattr(models, imputation_model_type)
            pretrained_model = SupervisedClass(**model_config)
            if type(pretrained_model) != LogisticRegression:
                pretrained_model.model = pretrained_model.model.to('cpu')

            no_pretraining_model = copy.deepcopy(pretrained_model)

            # https://github.com/facebookresearch/higher/pull/15
            gc.collect()

            subset_percent = subset_number * .1

            pretrain_list = [labeled_splits[(i+1) % 5], labeled_splits[(i+2) % 5]]
            train_list = [labeled_splits[(i+3) % 5], labeled_splits[(i+4) % 5]]
            val_data = labeled_splits[i]

            DatasetClass = type(labeled_data)
            pretrain_data = DatasetClass.from_list(train_list)
            train_data = DatasetClass.from_list(train_list)

            # Ensure the labels are encoded the same way in all three datasets
            pretrain_data.set_label_encoder(label_encoder)
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

            # TODO modify save path for no_pretraining_model

            neptune_run = None
            # Parse config file
            if args.neptune_config is not None:
                with open(args.neptune_config) as neptune_file:
                    neptune_config = yaml.safe_load(neptune_file)
                    neptune_run = utils.initialize_neptune(neptune_config)

            # Pretrain model on first split
            if torch.cuda.is_available() and type(pretrained_model) != LogisticRegression:
                pretrained_model.model = pretrained_model.model.to('cuda')
            pretrained_model.fit(pretrain_data, neptune_run)

            # Train pretrained model on second split
            model_type = ['pretrained', 'not_pretrained']
            for m_type, model in zip(model_type, [pretrained_model, no_pretraining_model]):
                neptune_run = None
                # Parse config file
                if args.neptune_config is not None:
                    with open(args.neptune_config) as neptune_file:
                        neptune_config = yaml.safe_load(neptune_file)
                        neptune_run = utils.initialize_neptune(neptune_config)

                model.fit(train_data, neptune_run)

                if torch.cuda.is_available() and type(model) != LogisticRegression:
                    model.model = model.model.to('cuda')

                predictions, true_labels = model.evaluate(val_data)

                if type(model) != LogisticRegression:
                    model.model = model.model.to('cpu')
                model.free_memory()

                accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
                balanced_acc = sklearn.metrics.balanced_accuracy_score(true_labels, predictions)

                accuracies.append(accuracy)
                balanced_accuracies.append(balanced_acc)
                train_studies.append(','.join(train_data.get_studies()))
                train_sample_names.append(','.join(train_data.get_samples()))
                val_sample_names.append(','.join(val_data.get_samples()))
                train_sample_counts.append(len(train_data))
                subset_percents.append(subset_percent)
                pretrained_or_trained.append(m_type)

            pretrain_data.reset_filters()
            train_data.reset_filters()

    with open(args.out_file, 'w') as out_file:
        # Write header
        out_file.write('accuracy\tbalanced_accuracy\ttrain studies\ttrain samples\t'
                       'val samples\ttrain sample count\tfraction of data used\tis_pretrained\n')

        result_iterator = zip(accuracies,
                              balanced_accuracies,
                              train_studies,
                              train_sample_names,
                              val_sample_names,
                              train_sample_counts,
                              subset_percents,
                              pretrained_or_trained
                              )
        for stats in result_iterator:
            stat_strings = [str(item) for item in stats]
            out_str = '\t'.join(stat_strings)
            out_file.write(f'{out_str}\n')
