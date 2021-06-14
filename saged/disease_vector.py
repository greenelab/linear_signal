"""
This script will jointly simulate data from healthy and disease data. It will then calculate
the centroid of the disease samples in the embedded space and use that centroid to extrapolate
datapoints to create easier and harder simulated classification problems
"""

import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from keras import backend as K
from ponyo import vae
from sklearn import preprocessing

from saged import datasets

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_config',
                        help='The config file containing info about the dataset')
    parser.add_argument('compendium_metadata',
                        help='A json file containing information compendium_file')
    parser.add_argument('out_dir', default='data/simulated',
                        help='The directory to store the simulated data files in')

    parser.add_argument('--simulation_config', default='model_configs/simulation/tybalt.yml')
    parser.add_argument('--sample_count', default=1000, type=int,
                        help='The number of samples to generate for each translated dataset')

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

    # Load healthy + disease data
    expression_data = labeled_data.subset_samples_to_labels([args.label,
                                                             args.negative_class])

    # Correct for batch effects
    if args.batch_correction_method is not None:
        expression_data = datasets.correct_batch_effects(expression_data,
                                                         args.batch_correction_method)

    # Scale data
    labeled_scaler = preprocessing.MinMaxScaler()
    train_data, _ = expression_data.get_all_data()
    labeled_scaler.fit(train_data)

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

    # Train a VAE on the data
    expression_array, _ = expression_data.get_all_data()
    expression_array = labeled_scaler.transform(expression_array)

    sample_ids = expression_data.get_samples()

    metadata['sample_ids'] = sample_ids

    expression_df = pd.DataFrame(expression_array, index=sample_ids)

    sample_to_label = expression_data.sample_to_label

    samples = expression_df.index
    labels = samples.map(sample_to_label)

    disease_df = expression_df[labels.str.match(args.label)]
    healthy_df = expression_df[labels.str.match(args.negative_class)]

    expression_encoder, expression_decoder, hist = vae.run_tybalt_training(expression_df,
                                                                           learning_rate,
                                                                           batch_size,
                                                                           epochs,
                                                                           kappa,
                                                                           intermediate_dim,
                                                                           latent_dim,
                                                                           epsilon_std,
                                                                           val_frac,)

    # Calculate the centroids for the embeddings of the disease and healthy samples
    disease_embeddings = expression_encoder.predict_on_batch(disease_df)
    healthy_embeddings = expression_encoder.predict_on_batch(healthy_df)

    disease_centroid = disease_embeddings.mean(axis=0)
    healthy_centroid = healthy_embeddings.mean(axis=0)
    disease_std = disease_embeddings.std(axis=0)
    healthy_std = healthy_embeddings.std(axis=0)

    print(disease_centroid)
    print(healthy_centroid)
    print(disease_std)
    print(healthy_std)

    # Sample disease data
    full_sample = np.zeros((args.sample_count, disease_embeddings.shape[1]))
    for i in range(args.sample_count):
        full_sample[i, :] = np.random.normal(loc=disease_centroid,
                                             scale=disease_std)

    sim_disease = expression_decoder.predict_on_batch(full_sample)

    out_path = os.path.join(args.out_dir, 'joint_sim_disease.pkl')
    with open(out_path, 'wb') as out_file:
        pickle.dump(sim_disease.T, out_file)

    # TODO save result

    disease_healthy_difference = disease_centroid - healthy_centroid
    for interpolation_amount in range(0, 11, 2):
        # Calculate the shifted centroid
        sim_centroid = healthy_centroid + (interpolation_amount * .1 * disease_healthy_difference)

        # Is there a good way to interpolate the standard deviation for each LV?
        # Linear interpolation seems dicier for scale than for location, but that may just be me
        sim_std = healthy_std

        # Sample data from the new centroid
        full_sample = np.zeros((args.sample_count, disease_embeddings.shape[1]))
        for i in range(args.sample_count):
            full_sample[i, :] = np.random.normal(loc=sim_centroid,
                                                 scale=sim_std)

        sim_data = expression_decoder.predict_on_batch(full_sample)

        # Store data
        out_path = os.path.join(args.out_dir,
                                'joint_sim_{:.1f}.pkl'.format(interpolation_amount * .1))
        with open(out_path, 'wb') as out_file:
            pickle.dump(sim_data.T, out_file)
