import argparse
import copy
import json
import os
import pickle
import random


import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from ponyo import vae
from keras import backend as K
from ponyo.simulate_expression_data import run_sample_simulation
from sklearn import preprocessing

from saged import utils, datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_config',
                        help='The config file containing info about the dataset')
    parser.add_argument('compendium_metadata',
                        help='A json file containing information compendium_file')
    parser.add_argument('out_dir', default='data/simulated',
                        help='The directory to store the simulated data files in')

    parser.add_argument('--simulation_config', default='model_configs/simulation/tybalt.yml')
    parser.add_argument('--sample_count', default=10000, type=int,
                        help='The number of samples of each class to generate')

    parser.add_argument('--label',
                        help='The label of the disease to be simulated',
                        default='sepsis')
    parser.add_argument('--negative_class',
                        help='The baseline class simulate. For example '
                             'in the refinebio dataset the negative class is "healthy"',
                        default='healthy')
    parser.add_argument('--batch_correction_method',
                        help='The method to use to correct for batch effects in the source data',
                        default=None)
    parser.add_argument('--seed', help='The number used to seed the RNG', default=42, type=int)
    parser.add_argument('--all_healthy',
                        help='Include all healthy samples, not just the ones in the same '
                        'study as the disease samples',
                        default=False,
                        action='store_true')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    metadata = {}

    with open(args.dataset_config) as data_file:
        dataset_config = yaml.safe_load(data_file)

    with open(args.simulation_config) as data_file:
        simulation_config = yaml.safe_load(data_file)

    all_data, labeled_data, unlabeled_data = datasets.load_binary_data(args.dataset_config,
                                                                       args.label,
                                                                       args.negative_class)

    if args.all_healthy:
        disease_data = labeled_data.subset_samples_to_labels([args.label,
                                                              args.negative_class])
    else:
        disease_samples = copy.deepcopy(labeled_data).subset_samples_to_labels([args.label])

        # Find which studies have data for the disease of interest
        sample_to_study = all_data.get_samples_to_studies()
        disease_studies = disease_samples.get_studies()
        all_labeled_samples = labeled_data.get_samples()

        # Get all samples in those studies
        disease_experiment_samples = utils.get_samples_in_studies(all_labeled_samples,
                                                                  disease_studies,
                                                                  sample_to_study)

        # Create a LabeledDataset object with those samples
        disease_data = labeled_data.subset_to_samples(disease_experiment_samples)

    # Correct for batch effects
    if args.batch_correction_method is not None:
        disease_data = datasets.correct_batch_effects(disease_data,
                                                      args.batch_correction_method)

    all_disease_samples = disease_data.get_samples()

    # Hold out twenty percent of the data for use later
    disease_data = disease_data.subset_studies(fraction=.8)

    labeled_scaler = preprocessing.MinMaxScaler()
    train_data, _ = disease_data.get_all_data()
    labeled_scaler.fit(train_data)

    train_samples = set(disease_data.get_samples())
    disease_only_data = copy.deepcopy(disease_data).subset_samples_to_labels([args.label])

    data_copy = copy.deepcopy(disease_data)
    healthy_only_data = data_copy.subset_samples_to_labels([args.negative_class])

    # Determine the held out samples
    held_out_samples = [sample for sample in all_disease_samples if sample not in train_samples]

    learning_rate = simulation_config['lr']
    batch_size = simulation_config['batch_size']
    epochs = simulation_config['epochs']
    kappa = simulation_config['kappa']
    intermediate_dim = simulation_config['intermediate_dim']
    latent_dim = simulation_config['latent_dim']
    epsilon_std = simulation_config['epsilon_std']
    val_frac = simulation_config['val_frac']

    # Set tf to be deterministic
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )

    # Create tf session
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # Train a VAE on disease
    disease_array, _ = disease_only_data.get_all_data()
    healthy_array, _ = healthy_only_data.get_all_data()
    disease_array = labeled_scaler.transform(disease_array)
    healthy_array = labeled_scaler.transform(healthy_array)

    disease_ids = disease_only_data.get_samples()
    healthy_ids = healthy_only_data.get_samples()

    metadata['disease_ids'] = disease_ids
    metadata['healthy_ids'] = healthy_ids

    disease_df = pd.DataFrame(disease_array, index=disease_ids)
    healthy_df = pd.DataFrame(healthy_array, index=healthy_ids)

    disease_encoder, disease_decoder, hist = vae.run_tybalt_training(disease_df,
                                                                     learning_rate,
                                                                     batch_size,
                                                                     epochs,
                                                                     kappa,
                                                                     intermediate_dim,
                                                                     latent_dim,
                                                                     epsilon_std,
                                                                     val_frac,)

    disease_simulated = run_sample_simulation(disease_encoder,
                                              disease_decoder,
                                              disease_df,
                                              args.sample_count)

    metadata['disease_loss'] = hist.history['loss']
    metadata['disease_val_loss'] = hist.history['val_loss']

    # Train a VAE on healthy
    healthy_encoder, healthy_decoder, hist = vae.run_tybalt_training(healthy_df,
                                                                     learning_rate,
                                                                     batch_size,
                                                                     epochs,
                                                                     kappa,
                                                                     intermediate_dim,
                                                                     latent_dim,
                                                                     epsilon_std,
                                                                     val_frac,)

    metadata['healthy_loss'] = hist.history['loss']
    metadata['healthy_val_loss'] = hist.history['val_loss']

    healthy_simulated = run_sample_simulation(healthy_encoder,
                                              healthy_decoder,
                                              healthy_df,
                                              args.sample_count)

    # Load all data not in healthy/disease
    used_samples = set(disease_ids + healthy_ids)
    all_samples = all_data.get_samples()
    all_unlabeled_samples = [sample for sample in all_samples if sample not in used_samples]
    metadata['unlabeled_ids'] = all_unlabeled_samples

    unlabeled_data = all_data.subset_to_samples(all_unlabeled_samples)
    unlabeled_array = unlabeled_data.get_all_data()

    unlabeled_scaler = preprocessing.MinMaxScaler()
    unlabeled_array = unlabeled_scaler.fit_transform(unlabeled_array)

    unlabeled_ids = unlabeled_data.get_samples()
    unlabeled_df = pd.DataFrame(unlabeled_array, index=unlabeled_ids)

    # train VAE on all data
    unlabeled_encoder, unlabeled_decoder, hist = vae.run_tybalt_training(unlabeled_df,
                                                                         learning_rate,
                                                                         batch_size,
                                                                         epochs,
                                                                         kappa,
                                                                         intermediate_dim,
                                                                         latent_dim,
                                                                         epsilon_std,
                                                                         val_frac,)

    metadata['unlabeled_loss'] = hist.history['loss']
    metadata['unlabeled_val_loss'] = hist.history['val_loss']

    # Generate unlabeled data from distribution as a whole
    unlabeled_simulated = run_sample_simulation(unlabeled_encoder,
                                                unlabeled_decoder,
                                                unlabeled_df,
                                                args.sample_count)

    metadata_file = os.path.join(args.out_dir,
                                 '{}_simulation_metadata.json'.format(args.label))
    labeled_scaler_file = os.path.join(args.out_dir,
                                       '{}_labeled_scaler.pkl'.format(args.label))
    unlabeled_scaler_file = os.path.join(args.out_dir,
                                         '{}_unlabeled_scaler.pkl'.format(args.label))

    healthy_out = os.path.join(args.out_dir,
                               '{}_train.pkl'.format(args.negative_class))
    disease_out = os.path.join(args.out_dir,
                               '{}_train.pkl'.format(args.label))

    with open(labeled_scaler_file, 'wb') as out_file:
        pickle.dump(labeled_scaler, out_file)
    with open(unlabeled_scaler_file, 'wb') as out_file:
        pickle.dump(unlabeled_scaler, out_file)

    with open(healthy_out, 'wb') as out_file:
        pickle.dump(healthy_array, out_file)
    with open(disease_out, 'wb') as out_file:
        pickle.dump(disease_array, out_file)

    metadata['train_samples'] = list(train_samples)
    metadata['test_samples'] = held_out_samples
    metadata['seed'] = args.seed
    metadata['disease'] = args.label
    metadata['healthy'] = args.negative_class

    with open(metadata_file, 'w') as out_file:
        json.dump(metadata, out_file)

    healthy_out = os.path.join(args.out_dir, '{}_sim.tsv'.format(args.negative_class))
    np.savetxt(healthy_out, healthy_simulated, delimiter='\t')
    disease_out = os.path.join(args.out_dir, '{}_sim.tsv'.format(args.label))
    np.savetxt(disease_out, disease_simulated, delimiter='\t')
    out_file_name = '{}_{}_unlabeled_sim.tsv'.format(args.label, args.negative_class)
    unlabeled_out = os.path.join(args.out_dir, out_file_name)
    np.savetxt(unlabeled_out, unlabeled_simulated, delimiter='\t')
