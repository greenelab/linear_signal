"""
This experiment determines whether models pretrained on data from different tissues than they
are predicting do better than models without pretraining
"""
import argparse
import copy
import gc
import random

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
    random.seed(args.seed)

    with open(args.dataset_config) as data_file:
        dataset_config = yaml.safe_load(data_file)

    expression_df, sample_to_label, sample_to_study = utils.load_recount_data(args.dataset_config)
    all_data = datasets.RefineBioMixedDataset(expression_df, sample_to_label, sample_to_study)

    labeled_data = all_data.get_labeled()

    # TODO should I reset the final layer like in standard transfer learning for the
    #      second training run? Probably, right?
    labeled_data.subset_samples_to_labels(PREDICT_TISSUES)
    labeled_data.recode()
    label_encoder = labeled_data.get_label_encoder()

    train_studies = []
    train_sample_names = []
    val_sample_names = []
    train_sample_counts = []
    subset_percents = []
    val_losses = []

    # Load model config
    input_size = all_data[0].shape[0]
    del(all_data)
    with open(args.model_config) as supervised_file:
        model_config = yaml.safe_load(supervised_file)
        model_config['input_size'] = input_size
        model_config['output_size'] = input_size

    accuracies = []
    balanced_accuracies = []
    pretrain_sample_names = []
    train_studies = []
    train_sample_names = []
    val_sample_names = []
    train_sample_counts = []
    subset_percents = []
    pretrained_or_trained = []

    for order in range(2):
        if order == 0:
            pretrain_data, train_data = utils.split_by_tissue(labeled_data,
                                                              PREDICT_TISSUES,
                                                              2)
        else:
            train_data, pretrain_data = utils.split_by_tissue(labeled_data,
                                                              PREDICT_TISSUES,
                                                              2)

        labeled_splits = train_data.get_cv_splits(num_splits=5,
                                                  seed=args.seed)

        # Pretrain a model once
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
        pretrain_data.set_label_encoder(label_encoder)

        # Ensure the model is training on GPU if possible
        if torch.cuda.is_available() and type(pretrain_base) != LogisticRegression:
            pretrain_base.model = pretrain_base.model.to('cuda')
        pretrain_base.fit(pretrain_data, neptune_run)

        if neptune_run is not None:
            neptune_run.close()

        # Move model back to CPU to allow easy copying
        if type(pretrain_base) != LogisticRegression:
            pretrain_base.model = pretrain_base.model.to('cpu')

        for i in range(len(labeled_splits)):
            for subset_number in range(1, 11, 1):

                # Grab a pretrained model and a copy of the original initialization
                pretrained_model = copy.deepcopy(pretrain_base)
                no_pretraining_model = copy.deepcopy(root_model)

                # https://github.com/facebookresearch/higher/pull/15
                gc.collect()

                subset_percent = subset_number * .1

                # Split = 50% pretrain, 40% train, 10% val
                train_list = labeled_splits[:i] + labeled_splits[i+1:]
                val_data = labeled_splits[i]

                DatasetClass = type(labeled_data)
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

                # TODO modify save path for no_pretraining_model

                neptune_run = None
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

                    if neptune_run is not None:
                        neptune_run.close()

                    if torch.cuda.is_available() and type(model) != LogisticRegression:
                        model.model = model.model.to('cuda')

                    predictions, true_labels = model.evaluate(val_data)

                    if type(model) != LogisticRegression:
                        model.model = model.model.to('cpu')
                    model.free_memory()

                    accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
                    balanced_acc = sklearn.metrics.balanced_accuracy_score(true_labels,
                                                                           predictions)

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
