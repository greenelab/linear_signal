"""
This benchmark compares the performance of different models in
predicting tissue based on gene expression
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import sklearn.metrics
import yaml
from sklearn.preprocessing import MinMaxScaler

from saged import utils, datasets, models

# Note, the arguments will have underscores, but the labels in the encoder
# will use spaces
AVAILABLE_TISSUES = ['Blood', 'Breast', 'Stem_Cell', 'Cervix', 'Brain', 'Kidney',
                     'Umbilical_Cord', 'Lung', 'Epithelium', 'Prostate', 'Liver',
                     'Heart', 'Skin', 'Colon', 'Bone Marrow', 'Muscle', 'Tonsil', 'Blood_Vessel',
                     'Spinal_Cord', 'Testis', 'Placenta', 'Bladder', 'Adipose_Tisse', 'Ovary',
                     'Melanoma', 'Adrenal_Gland', 'Bone', 'Pancreas', 'Penis',
                     'Universal reference', 'Spleen', 'Brain_reference', 'Large_Intestine',
                     'Esophagus', 'Small Intestine', 'Embryonic_kidney', 'Thymus', 'Stomach',
                     'Endometrium', 'Glioblastoma', 'Gall_bladder', 'Lymph_Nodes', 'Airway',
                     'Appendix', 'Thyroid', 'Retina', 'Bowel_tissue', 'Foreskin', 'Sperm', 'Foot',
                     'Cerebellum', 'Cerebral_cortex', 'Salivary_Gland', 'Duodenum'
                     ]

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
    parser.add_argument('--tissue1',
                        help='The first tissue to be predicted from the data',
                        default='Blood', choices=AVAILABLE_TISSUES)
    parser.add_argument('--tissue2',
                        help='The second tissue to be predicted from the data',
                        default='Breast', choices=AVAILABLE_TISSUES)
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
    parser.add_argument('--all_tissue', help='Predict all common tissues in the dataset',
                        default=False, action='store_true')
    parser.add_argument('--biobert', help='Add biobert embeddings as features the model can use',
                        default=False, action='store_true')
    parser.add_argument('--weighted_loss',
                        help='Weight classes based on the inverse of their prevalence',
                        action='store_true')
    parser.add_argument('--signal_removal',
                        help='If this flag is set, limma will be used to remove linear signals '
                             'associated with the labels',
                        action='store_true')
    parser.add_argument('--study_correct',
                        help='If this flag is set, limma will be used to remove linear signals '
                             'associated with the studies',
                        action='store_true')
    parser.add_argument('--use_sex_labels',
                        help='If this flag is set, use Flynn sex labels instead of tissue labels',
                        action='store_true')

    args = parser.parse_args()

    expression_df, sample_to_label, sample_to_study = utils.load_recount_data(args.dataset_config)

    with open(args.dataset_config) as in_file:
        dataset_config = yaml.safe_load(in_file)

        if args.use_sex_labels:
            label_path = dataset_config.pop('sex_label_path')
            sample_to_label = utils.parse_flynn_labels(label_path)

    if args.biobert:
        embeddings = utils.load_biobert_embeddings(args.dataset_config)

        # These indices are correct, the expression dataframe is genes x samples currently
        placeholder_array = np.ones((embeddings.shape[1], expression_df.shape[1]))
        with open(dataset_config['metadata_path'], 'r') as in_file:
            header = in_file.readline()
            header = header.replace('"', '')
            header = header.strip().split('\t')

            # Add one to the indices to account for the index column in metadata not present in
            # the header
            sample_index = header.index('external_id') + 1
            for line_number, metadata_line in enumerate(in_file):
                line = metadata_line.strip().split('\t')
                sample = line[sample_index]
                sample = sample.replace('"', '')

                # Not all samples with metadata are in compendium
                if sample not in expression_df.columns:
                    continue

                index_in_df = expression_df.columns.get_loc(sample)
                placeholder_array[:, index_in_df] = embeddings[line_number, :]

            # 0-1 normalize embeddings to match scale of expression
            scaler = MinMaxScaler()
            placeholder_array = scaler.fit_transform(placeholder_array.T).T

            embedding_df = pd.DataFrame(placeholder_array, columns=expression_df.columns)
            expression_df = pd.concat([expression_df, embedding_df], axis='rows')

    all_data = datasets.RefineBioMixedDataset(expression_df, sample_to_label, sample_to_study)

    labeled_data = all_data.get_labeled()

    labels_to_keep = None
    if args.all_tissue:
        # Keep all labels with at least ten studies in the dataset
        labels_to_keep = ['Blood', 'Breast', 'Stem Cell', 'Cervix', 'Brain', 'Kidney',
                          'Umbilical Cord', 'Lung', 'Epithelium', 'Prostate', 'Liver',
                          'Heart', 'Skin', 'Colon', 'Bone Marrow', 'Muscle', 'Tonsil',
                          'Blood Vessel', 'Spinal Cord', 'Testis', 'Placenta'
                          ]
    else:
        tissue_1 = args.tissue1.replace('_', ' ')
        tissue_2 = args.tissue2.replace('_', ' ')
        labels_to_keep = [tissue_1, tissue_2]

    if not args.use_sex_labels:
        labeled_data.subset_samples_to_labels(labels_to_keep)

    # Correct for batch effects
    if args.study_correct:
        labeled_data = datasets.correct_batch_effects(labeled_data, 'limma', 'studies')

    if args.signal_removal:
        labeled_data = datasets.correct_batch_effects(labeled_data, 'limma', 'labels')

    labeled_data.recode()
    label_encoder = labeled_data.get_label_encoder()

    # Get fivefold cross-validation splits
    print('CV splitting')
    labeled_splits = labeled_data.get_cv_splits(num_splits=args.num_splits, seed=args.seed)

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
    for i in range(len(labeled_splits)):
        for subset_number in range(1, 11, 1):
            # The new neptune version doesn't have a create_experiment function so you have to
            # reinitialize per-model
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

            # This isn't strictly necessary since we're checking whether both classes are present,
            # but it's safer
            train_data.set_label_encoder(label_encoder)
            val_data.set_label_encoder(label_encoder)

            # Now that the ratio is correct, actually subset the samples
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
                if args.weighted_loss:
                    loss_weights = utils.calculate_loss_weights(train_data)
                supervised_config['loss_weights'] = loss_weights
                if 'save_path' in supervised_config:
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

                    extra_info = '{}_{}_{}'.format(extra_info, i, args.seed)

                    save_path = os.path.join(save_path + '_predict_{}.pt'.format(extra_info))

                    supervised_config['save_path'] = save_path

            supervised_model_type = supervised_config.pop('name')
            SupervisedClass = getattr(models, supervised_model_type)
            supervised_model = SupervisedClass(**supervised_config)

            supervised_model.fit(train_data, neptune_run)

            predictions, true_labels = supervised_model.evaluate(val_data)

            supervised_model.free_memory()

            accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
            balanced_acc = sklearn.metrics.balanced_accuracy_score(true_labels, predictions)

            label_mapping = dict(zip(label_encoder.classes_,
                                     range(len(label_encoder.classes_))))

            # Ensure this mapping is correct
            for label in label_mapping.keys():
                assert label_encoder.transform([label]) == label_mapping[label]

            encoder_string = json.dumps(label_mapping)
            prediction_string = ','.join(list(predictions.astype('str')))
            truth_string = ','.join(list(true_labels.astype('str')))

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

            train_data.reset_filters()
            val_data.reset_filters()

    with open(args.out_file, 'w') as out_file:
        # Write header
        out_file.write('accuracy\tbalanced_accuracy\ttrain studies\ttrain samples\t'
                       'val samples\ttrain sample count\tfraction of data used\t'
                       'val_predictions\tval_true_labels\tval_encoders\n')

        result_iterator = zip(accuracies,
                              balanced_accuracies,
                              supervised_train_studies,
                              supervised_train_sample_names,
                              supervised_val_sample_names,
                              supervised_train_sample_counts,
                              subset_percents,
                              val_predictions,
                              val_true_labels,
                              val_encoders
                              )
        for stats in result_iterator:
            stat_strings = [str(item) for item in stats]
            out_str = '\t'.join(stat_strings)
            out_file.write(f'{out_str}\n')
