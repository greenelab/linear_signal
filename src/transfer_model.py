"""
This benchmark compares the performance of different models in
predicting tissue based on gene expression
"""
import argparse
import copy
import json
import os
from typing import Tuple

import optuna
import pandas as pd
import sklearn.metrics
import torch
import yaml

import utils
import datasets
import models

# Note, the arguments will have underscores, but the labels in the encoder
# will use spaces
AVAILABLE_TISSUES = ['Blood', 'Breast', 'Stem_Cell', 'Cervix', 'Brain', 'Kidney',
                     'Umbilical_Cord', 'Lung', 'Epithelium', 'Prostate', 'Liver',
                     'Heart', 'Skin', 'Colon', 'Bone_Marrow', 'Muscle', 'Tonsil', 'Blood_Vessel',
                     'Spinal_Cord', 'Testis', 'Placenta', 'Bladder', 'Adipose_Tisse', 'Ovary',
                     'Melanoma', 'Adrenal_Gland', 'Bone', 'Pancreas', 'Penis',
                     'Universal_reference', 'Spleen', 'Brain_reference', 'Large_Intestine',
                     'Esophagus', 'Small_Intestine', 'Embryonic_kidney', 'Thymus', 'Stomach',
                     'Endometrium', 'Glioblastoma', 'Gall_bladder', 'Lymph_Nodes', 'Airway',
                     'Appendix', 'Thyroid', 'Retina', 'Bowel_tissue', 'Foreskin', 'Sperm', 'Foot',
                     'Cerebellum', 'Cerebral_cortex', 'Salivary_Gland', 'Duodenum'
                     ]

SHARED_TISSUES = ['Blood', 'Brain', 'Skin', 'Blood Vessel', 'Heart', 'Muscle',
                  'Lung', 'Colon', 'Breast', 'Prostate', 'Liver', 'Kidney']


def objective(trial, train_list, supervised_config,
              label_encoder, weighted_loss=False, device='cpu'):
    losses = []

    with open(supervised_config) as supervised_file:
        supervised_config = yaml.safe_load(supervised_file)
    supervised_model_type = supervised_config.pop('name')

    if supervised_model_type != 'LogisticRegression':
        lr = trial.suggest_float('lr', 1e-6, 10, log=True)
    l2_penalty = trial.suggest_float('l2_penalty', 1e-6, 10, log=True)

    for i in range(len(train_list)):
        inner_train_list = train_list[:i] + train_list[i+1:]
        LabeledDatasetClass = type(labeled_data)

        inner_train_data = LabeledDatasetClass.from_list(inner_train_list)
        inner_val_data = train_list[i]
        inner_train_data.set_label_encoder(label_encoder)
        inner_val_data.set_label_encoder(label_encoder)

        # Sklearn logistic regression doesn't allow manually specifying classes
        # so we have to do this
        if len(inner_train_data.get_classes()) < len(inner_val_data.get_classes()):
            continue

        input_size = len(inner_train_data.get_features())
        output_size = len(label_encoder.classes_)

        with open(args.supervised_config) as supervised_file:
            supervised_config = yaml.safe_load(supervised_file)
            supervised_config['input_size'] = input_size
            supervised_config['output_size'] = output_size
            supervised_config['log_progress'] = False
            supervised_config['l2_penalty'] = l2_penalty
            if supervised_model_type != 'LogisticRegression':
                supervised_config['lr'] = lr
            if weighted_loss:
                loss_weights = utils.calculate_loss_weights(inner_train_data)
                supervised_config['loss_weights'] = loss_weights

        supervised_model_type = supervised_config.pop('name')
        SupervisedClass = getattr(models, supervised_model_type)
        supervised_model = SupervisedClass(**supervised_config)

        supervised_model.fit(inner_train_data)

        _, true_labels = supervised_model.evaluate(inner_val_data)
        outputs = supervised_model.predict_proba(inner_val_data)

        loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)
        out_tensor = torch.tensor(outputs, dtype=torch.float).to(device)
        label_tensor = torch.tensor(true_labels, dtype=torch.long).to(device)
        loss = loss_fn(out_tensor, label_tensor)

        losses.append(loss)

        supervised_model.free_memory()

    return sum(losses) / len(losses)


def prep_recount_data(config, args: argparse.Namespace) -> Tuple[datasets.RefineBioMixedDataset,
                                                                 datasets.RefineBioLabeledDataset]:
    expression_df, sample_to_label, sample_to_study = utils.load_recount_data(config)

    with open(config) as in_file:
        dataset_config = yaml.safe_load(in_file)

        if args.use_sex_labels:
            label_path = dataset_config.pop('sex_label_path')
            sample_to_label = utils.parse_flynn_labels(label_path)

    all_data = datasets.RefineBioMixedDataset(expression_df, sample_to_label, sample_to_study)

    labeled_data = all_data.get_labeled()

    labels_to_keep = None
    if args.all_tissue:
        # Keep all labels with at least ten studies in the dataset
        labels_to_keep = SHARED_TISSUES
    else:
        tissue_1 = args.tissue1.replace('_', ' ')
        tissue_2 = args.tissue2.replace('_', ' ')
        labels_to_keep = [tissue_1, tissue_2]

    if not args.use_sex_labels:
        labeled_data.subset_samples_to_labels(labels_to_keep)

    return all_data, labeled_data


def prep_gtex_data(config, args: argparse.Namespace) -> Tuple[datasets.RefineBioMixedDataset,
                                                              datasets.RefineBioLabeledDataset]:
    # Load dataset config
    with open(config) as in_file:
        dataset_config = yaml.safe_load(in_file)
    metadata_path = dataset_config['metadata_path']

    expression_df = utils.load_compendium_file(dataset_config['compendium_path']).T

    sample_to_study = utils.get_gtex_sample_to_study(metadata_path)
    sample_to_label = utils.get_gtex_sample_to_label(metadata_path)
    # Create MixedDataset
    all_data = datasets.RefineBioMixedDataset(expression_df, sample_to_label, sample_to_study)

    labeled_data = all_data.get_labeled()

    if args.all_tissue:
        # Keep all labels with at least ten studies in the dataset
        labels_to_keep = SHARED_TISSUES
    else:
        tissue_1 = args.tissue1.replace('_', ' ')
        tissue_2 = args.tissue2.replace('_', ' ')
        labels_to_keep = [tissue_1, tissue_2]

    labeled_data.subset_samples_to_labels(labels_to_keep)

    return all_data, labeled_data


def prep_sim_data(args: argparse.Namespace) -> Tuple[datasets.RefineBioMixedDataset,
                                                     datasets.RefineBioLabeledDataset]:
    with open(args.dataset_config) as in_file:
        dataset_config = yaml.safe_load(in_file)

    data_df = pd.read_csv(dataset_config['compendium_path'], sep='\t', index_col=0)
    sample_to_label = dict(zip(data_df.index, data_df['label']))

    sample_to_study = {sample: sample + '_study' for sample in data_df.index}

    data_df = data_df.drop(['label'], axis='columns')
    data_df = data_df.T

    all_data = datasets.RefineBioMixedDataset(data_df, sample_to_label, sample_to_study)

    labeled_data = all_data.get_labeled()

    return all_data, labeled_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_config',
                        help='The yaml formatted dataset configuration file. For more information '
                             'about this file read the comments in the example_dataset.yml file')
    parser.add_argument('transfer_data_config',
                        help='The dataset config for the data the model will be transfered to')
    parser.add_argument('supervised_config',
                        help='The yaml formatted model configuration file. For more information '
                             'about this file read the comments in the example_model.yml file')
    parser.add_argument('out_file',
                        help='The file to save the results to')
    parser.add_argument('--dataset',
                        help='The dataset to be used as the training data',
                        choices=['gtex', 'recount'],
                        default='recount')
    parser.add_argument('--neptune_config',
                        help='A yaml formatted file containing init information for '
                             'neptune logging')
    parser.add_argument('--seed',
                        help='The random seed to be used in splitting data',
                        type=int,
                        default=42)
    parser.add_argument('--num_splits',
                        help='The number of splits to use in cross-validation (must be at least 3)',
                        type=int,
                        default=5)
    parser.add_argument('--weighted_loss',
                        help='Weight classes based on the inverse of their prevalence',
                        action='store_true')
    parser.add_argument('--correction',
                        help='This argument determines how signal correction will be run.'
                             'If "signal", then all linear signal associated with the labels '
                             'will be removed.'
                             'If "study", all linear study signal will be removed'
                             'If "split_signal", all linear signal will be removed separately '
                             'for the train and val sets',
                        choices=['uncorrected', 'signal', 'study', 'split_signal'],
                        default='uncorrected')
    parser.add_argument('--sample_split',
                        help='If this flag is set, split cv folds at the sample level instead '
                             'of the study level',
                        action='store_true')
    parser.add_argument('--disable_optuna',
                        help="If this flag is set, don't to hyperparameter optimization",
                        action='store_true')
    # Recount/GTEX args
    parser.add_argument('--tissue1',
                        help='The first tissue to be predicted from the data',
                        default='Blood', choices=AVAILABLE_TISSUES)
    parser.add_argument('--tissue2',
                        help='The second tissue to be predicted from the data',
                        default='Breast', choices=AVAILABLE_TISSUES)
    parser.add_argument('--all_tissue', help='Predict all common tissues in the dataset',
                        default=False, action='store_true')
    # Recount only args
    parser.add_argument('--biobert', help='Add biobert embeddings as features the model can use',
                        default=False, action='store_true')
    parser.add_argument('--use_sex_labels',
                        help='If this flag is set, use Flynn sex labels instead of tissue labels',
                        action='store_true')

    args = parser.parse_args()

    if args.num_splits < 3:
        raise ValueError('The num_splits argument must be >= 3')

    if args.dataset == 'recount':
        all_data, labeled_data = prep_recount_data(args.train_data_config, args)
        _, transfer_data = prep_gtex_data(args.transfer_data_config, args)
    elif args.dataset == 'gtex':
        all_data, labeled_data = prep_gtex_data(args.train_data_config, args)
        _, transfer_data = prep_recount_data(args.transfer_data_config, args)

    # Correct for batch effects
    if args.correction == 'study':
        labeled_data = datasets.correct_batch_effects(labeled_data, 'limma', 'studies')
    elif args.correction == 'signal':
        labeled_data = datasets.correct_batch_effects(labeled_data, 'limma', 'labels')

    labeled_data.recode()
    label_encoder = labeled_data.get_label_encoder()

    # Get fivefold cross-validation splits
    print('CV splitting')
    labeled_splits = labeled_data.get_cv_splits(num_splits=args.num_splits,
                                                seed=args.seed,
                                                split_by_sample=args.sample_split)

    # Train the model on each fold
    accuracies = []
    balanced_accuracies = []
    supervised_train_studies = []
    supervised_train_sample_names = []
    supervised_val_sample_names = []
    supervised_train_sample_counts = []
    subset_percents = []
    val_predictions = []
    val_true_labels = []
    val_encoders = []
    transfer_predictions = []
    transfer_balanced_accuracies = []

    tune_data = labeled_splits[0]

    for i in range(len(labeled_splits)):
        # Select hyperparameters via nested CV
        train_list = labeled_splits[:i] + labeled_splits[i+1:]

        if not args.disable_optuna:
            # optuna.logging.set_verbosity(optuna.logging.ERROR)
            sampler = optuna.samplers.RandomSampler(seed=args.seed)
            study = optuna.create_study()
            print('Tuning hyperparameters...')

            study.optimize(lambda trial: objective(trial,
                                                   train_list,
                                                   args.supervised_config,
                                                   label_encoder,
                                                   args.weighted_loss,
                                                   ),
                           n_trials=25,
                           show_progress_bar=True)

        # The new neptune version doesn't have a create_experiment function so you have to
        # reinitialize per-model
        neptune_run = None
        # Parse config file
        if args.neptune_config is not None:
            with open(args.neptune_config) as neptune_file:
                neptune_config = yaml.safe_load(neptune_file)
                neptune_run = utils.initialize_neptune(neptune_config)

        subset_percent = 1

        train_list = labeled_splits[:i] + labeled_splits[i+1:]

        # Extract the train and test datasets
        LabeledDatasetClass = type(labeled_data)
        train_data = LabeledDatasetClass.from_list(train_list)
        val_data = labeled_splits[i]

        # Ensure the labels are the same in all three datasets
        train_data.set_label_encoder(label_encoder)
        val_data.set_label_encoder(label_encoder)
        transfer_data.set_label_encoder(label_encoder)

        if args.correction == 'split_signal':
            # Using deep copies to make sure data resets to its original state each iteration
            train_data = datasets.correct_batch_effects(copy.deepcopy(train_data),
                                                        'limma', 'labels')
            val_data = datasets.correct_batch_effects(copy.deepcopy(val_data),
                                                      'limma', 'labels')

        train_data = train_data.subset_samples(subset_percent,
                                               args.seed)

        # Skip entries where there is only data for one class
        if len(train_data.get_classes()) <= 1 or len(val_data.get_classes()) <= 1:
            continue

        if args.neptune_config is not None:
            neptune_run['samples'] = len(train_data.get_samples())
            neptune_run['studies'] = len(train_data.get_studies())

        print('Samples: {}'.format(len(train_data.get_samples())))
        print('Studies: {}'.format(len(train_data.get_studies())))

        print('Val data: {}'.format(len(val_data)))
        input_size = len(train_data.get_features())
        output_size = len(label_encoder.classes_)
        print('Classes: {}'.format(output_size))

        with open(args.supervised_config) as supervised_file:
            supervised_config = yaml.safe_load(supervised_file)
            supervised_config['input_size'] = input_size
            supervised_config['output_size'] = output_size

            # Use optimized values for lr etc if available
            if not args.disable_optuna:
                param_dict = study.best_params
                for param in param_dict.keys():
                    supervised_config[param] = param_dict[param]

            if args.weighted_loss:
                loss_weights = utils.calculate_loss_weights(train_data)
                supervised_config['loss_weights'] = loss_weights
            # Only save one model per run to avoid running out of disk space
            if 'save_path' in supervised_config and i == 0 and args.seed == 0:
                # Append script-specific information to model save file
                save_path = supervised_config['save_path']
                # Remove extension
                save_path = os.path.splitext(save_path)[0]

                if args.all_tissue and args.biobert:
                    extra_info = 'all_tissue_biobert'
                elif args.use_sex_labels:
                    extra_info = 'sex_prediction'
                elif args.all_tissue:
                    extra_info = 'all_tissue'
                elif args.biobert:
                    extra_info = 'biobert'
                else:
                    extra_info = '{}-{}'.format(args.tissue1, args.tissue2)

                extra_info = 'transfer_{}_{}_{}_{}_{}'.format(args.dataset, extra_info,
                                                              args.correction, i, args.seed)

                save_path = os.path.join(save_path + '_predict_{}.pt'.format(extra_info))

                supervised_config['save_path'] = save_path

        supervised_model_type = supervised_config.pop('name')
        SupervisedClass = getattr(models, supervised_model_type)
        supervised_model = SupervisedClass(**supervised_config)

        supervised_model.fit(train_data, neptune_run)

        predictions, true_labels = supervised_model.evaluate(val_data)
        transfer_pred, transfer_labels = supervised_model.evaluate(transfer_data)

        supervised_model.free_memory()

        accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
        balanced_acc = sklearn.metrics.balanced_accuracy_score(true_labels, predictions)
        transfer_balanced_acc = sklearn.metrics.balanced_accuracy_score(transfer_labels,
                                                                        transfer_pred)

        # The downstream json conversion hates number keys, so we'll make sure
        # everything is a string instead
        str_classes = [str(id) for id in label_encoder.classes_]
        label_mapping = dict(zip(str_classes,
                                 range(len(label_encoder.classes_))))

        encoder_string = json.dumps(label_mapping)
        prediction_string = ','.join(list(predictions.astype('str')))
        truth_string = ','.join(list(true_labels.astype('str')))
        transfer_prediction_string = ','.join(list(transfer_pred.astype('str')))

        accuracies.append(accuracy)
        balanced_accuracies.append(balanced_acc)
        supervised_train_studies.append(','.join(train_data.get_studies()))
        supervised_train_sample_names.append(','.join(train_data.get_samples()))
        supervised_val_sample_names.append(','.join(val_data.get_samples()))
        supervised_train_sample_counts.append(len(train_data))
        subset_percents.append(subset_percent)
        val_predictions.append(prediction_string)
        val_true_labels.append(truth_string)
        val_encoders.append(encoder_string)
        transfer_predictions.append(transfer_prediction_string)
        transfer_balanced_accuracies.append(transfer_balanced_acc)

        train_data.reset_filters()
        val_data.reset_filters()

    with open(args.out_file, 'w') as out_file:
        # Write header
        out_file.write('accuracy\tbalanced_accuracy\ttrain studies\ttrain samples\t'
                       'val samples\ttrain sample count\tfraction of data used\t'
                       'val_predictions\tval_true_labels\tval_encoders\t'
                       'transfer_predictions\ttransfer_balanced_accuracy\n')

        result_iterator = zip(accuracies,
                              balanced_accuracies,
                              supervised_train_studies,
                              supervised_train_sample_names,
                              supervised_val_sample_names,
                              supervised_train_sample_counts,
                              subset_percents,
                              val_predictions,
                              val_true_labels,
                              val_encoders,
                              transfer_predictions,
                              transfer_balanced_accuracies
                              )
        for stats in result_iterator:
            stat_strings = [str(item) for item in stats]
            out_str = '\t'.join(stat_strings)
            out_file.write(f'{out_str}\n')
