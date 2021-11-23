"""
This benchmark trains two models, one trained on half the labeled data then retrained on the other
half, and one only trained on the second half. This acts as a positive control showing that
pretraining has a positive effect on model accuracy under ideal conditions
"""
import argparse
import copy
import gc
import json
import os

import sklearn.metrics
import torch

import yaml

from saged import utils, datasets, models
from saged.models import LogisticRegression
from saged.utils import calculate_loss_weights


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
    parser.add_argument('--sample_split',
                        help='If this flag is set, split cv folds at the sample level instead '
                             'of the study level',
                        action='store_true')
    parser.add_argument('--weighted_loss',
                        help='Weight classes based on the inverse of their prevalence',
                        action='store_true')
    parser.add_argument('--sex_label_path',
                        help='The path to the labels from Flynn et al., if they should be used',
                        default=None)

    args = parser.parse_args()

    with open(args.dataset_config) as data_file:
        dataset_config = yaml.safe_load(data_file)

    expression_df, sample_to_label, sample_to_study = utils.load_recount_data(args.dataset_config)

    if 'sex_label_path' in args:
        sample_to_label = utils.parse_flynn_labels(args.sex_label_path)

    all_data = datasets.RefineBioMixedDataset(expression_df, sample_to_label, sample_to_study)

    labeled_data = all_data.get_labeled()
    if 'sex_label_path' not in args:
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

    # Split the data into two sets
    labeled_splits = labeled_data.get_cv_splits(num_splits=5,
                                                seed=args.seed,
                                                split_by_sample=args.sample_split)

    del(all_data)

    with open(args.model_config) as supervised_file:
        model_config = yaml.safe_load(supervised_file)
        model_config['input_size'] = input_size
        model_config['output_size'] = len(label_encoder.classes_)

    if args.weighted_loss:
        loss_weights = calculate_loss_weights(labeled_data)
        model_config['loss_weights'] = loss_weights

    accuracies = []
    balanced_accuracies = []
    pretrain_sample_names = []
    train_studies = []
    train_sample_names = []
    val_sample_names = []
    train_sample_counts = []
    subset_percents = []
    pretrained_or_trained = []
    val_predictions = []
    val_true_labels = []
    val_encoders = []

    for i in range(len(labeled_splits)):
        neptune_run = None
        # Parse config file
        if args.neptune_config is not None:
            with open(args.neptune_config) as neptune_file:
                neptune_config = yaml.safe_load(neptune_file)
                neptune_run = utils.initialize_neptune(neptune_config)

        model_type = model_config['name']
        SupervisedClass = getattr(models, model_type)
        root_model = SupervisedClass(**model_config)

        pretrain_base = copy.deepcopy(root_model)

        # Get the pretraining fraction of the data
        DatasetClass = type(labeled_data)
        pretrain_list = [labeled_splits[(i+1) % 5], labeled_splits[(i+2) % 5]]
        pretrain_data = DatasetClass.from_list(pretrain_list)
        pretrain_data.set_label_encoder(label_encoder)

        # Ensure the model is training on GPU if possible
        if torch.cuda.is_available() and type(pretrain_base) != LogisticRegression:
            pretrain_base.model = pretrain_base.model.to('cuda')
        pretrain_base.fit(pretrain_data, neptune_run)

        # Move model back to CPU to allow easy copying
        if type(pretrain_base) != LogisticRegression:
            pretrain_base.model = pretrain_base.model.to('cpu')

        for subset_number in range(1, 11, 1):

            # Grab a pretrained model and a copy of the original initialization for no_pretraining
            pretrained_model = copy.deepcopy(pretrain_base)
            no_pretraining_model = copy.deepcopy(root_model)

            # https://github.com/facebookresearch/higher/pull/15
            gc.collect()

            subset_percent = subset_number * .1

            # Split = 40% pretrain, 40% train, 20% val
            train_list = [labeled_splits[(i+3) % 5], labeled_splits[(i+4) % 5]]
            val_data = labeled_splits[i]

            train_data = DatasetClass.from_list(train_list)

            # Ensure the labels are encoded the same way in all three datasets
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

            neptune_run = None
            # Parse config file
            if args.neptune_config is not None:
                with open(args.neptune_config) as neptune_file:
                    neptune_config = yaml.safe_load(neptune_file)
                    neptune_run = utils.initialize_neptune(neptune_config)

            # Train or fine tune the model on the train set
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

                # Don't save every model, just a representative one per class
                if subset_number == 10 and args.seed == 0:
                    if 'save_path' in model_config:
                        model_save_path = model_config['save_path']
                        model_save_path = os.path.dirname(model_save_path)

                        # Sample or study split
                        if args.sample_split:
                            model_save_path += '/sample-level'
                        else:
                            model_save_path += '/study-level'

                        # Sex prediction or tissue prediction
                        if 'sex_label_path' in args:
                            model_save_path += '-sex-prediction'
                        model_save_path += '_'

                        # Model class
                        model_save_path += '{}_'.format(model_config['name'])

                        # Pretrained or not
                        model_save_path += '{}'.format(m_type)

                        model_save_path += '.pt'

                        model.save_model(model_save_path)

                if type(model) != LogisticRegression:
                    model.model = model.model.to('cpu')
                model.free_memory()

                accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
                balanced_acc = sklearn.metrics.balanced_accuracy_score(true_labels, predictions)

                label_mapping = dict(zip(label_encoder.classes_,
                                         range(len(label_encoder.classes_))))

                # Ensure this mapping is correct
                for label in label_mapping.keys():
                    assert label_encoder.transform([label]) == label_mapping[label]

                encoder_string = json.dumps(label_mapping)
                # Format predictions to be a comma separated string of numbers without spaces
                prediction_string = ','.join(list(predictions.astype('str')))
                truth_string = ','.join(list(true_labels.astype('str')))

                accuracies.append(accuracy)
                balanced_accuracies.append(balanced_acc)
                train_studies.append(','.join(train_data.get_studies()))
                train_sample_names.append(','.join(train_data.get_samples()))
                val_sample_names.append(','.join(val_data.get_samples()))
                train_sample_counts.append(len(train_data))
                subset_percents.append(subset_percent)
                pretrained_or_trained.append(m_type)
                val_predictions.append(prediction_string)
                val_true_labels.append(truth_string)
                val_encoders.append(encoder_string)

            pretrain_data.reset_filters()
            train_data.reset_filters()

    with open(args.out_file, 'w') as out_file:
        # Write header
        out_file.write('accuracy\tbalanced_accuracy\ttrain studies\ttrain samples\t'
                       'val samples\ttrain sample count\tfraction of data used\tis_pretrained\t'
                       'val_predictions\tval_true_labels\tval_encoders\n')

        result_iterator = zip(accuracies,
                              balanced_accuracies,
                              train_studies,
                              train_sample_names,
                              val_sample_names,
                              train_sample_counts,
                              subset_percents,
                              pretrained_or_trained,
                              val_predictions,
                              val_true_labels,
                              val_encoders
                              )
        for stats in result_iterator:
            stat_strings = [str(item) for item in stats]
            out_str = '\t'.join(stat_strings)
            out_file.write(f'{out_str}\n')
