""" This script creates the metatdata required for simulated data to be used in
RefineBioDataset objects """

import argparse
import glob
import json
import os
import pickle

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='The directory containing simulated data to generate'
                                         'metadata for')
    parser.add_argument('compendium_out_file', help='The location to store the '
                                                    'combined expression to')
    parser.add_argument('metadata_out_file', help='The location to save the metadata to')
    parser.add_argument('label_out_file', help='The location to save the label file to')

    args = parser.parse_args()

    sim_files = glob.glob(args.data_dir + '/*_sim.tsv')

    metadata = {'experiments': {}, 'samples': {}}


    all_data_df = None

    labeled_samples = set()
    label_to_samples = {}

    for sim_file in sim_files:
        file_name = os.path.basename(sim_file)
        current_df = pd.read_csv(os.path.join(args.data_dir, file_name),
                                 delimiter='\t', header=None)

        label = file_name.rstrip('_sim.tsv')

        sample_count = len(current_df.index)

        # Generate dummy experiment and sample ids for use in
        sample_ids = ['{}_{}'.format(label, i) for i in range(sample_count)]
        current_df.index = sample_ids
        experiment_ids = ['Experiment_{}_{}'.format(label, i) for i in range(sample_count)]

        label_to_samples[label] = sample_ids
        labeled_samples.update(sample_ids)

        # Write metadata in refinebio format for the samples
        for experiment_id, sample_id in zip(experiment_ids, sample_ids):
            metadata['experiments'][experiment_id] = {'sample_accession_codes': [sample_id]}
            sample_dict = {"platform": "simulation",
                           "refinebio_annotations": [{"characteristics_ch1": "tissue: blood"}]}
            metadata['samples'][sample_id] = sample_dict

        # Combine the dataframes into a compendium
        if all_data_df is None:
            all_data_df = current_df
        else:
            all_data_df = pd.concat([all_data_df, current_df], axis=0)

    # Save compendium
    all_data_df.to_pickle(args.compendium_out_file)

    # Save label file
    with open(args.label_out_file, 'wb') as out_file:
        pickle.dump((label_to_samples, labeled_samples), out_file)

    # Save metdata
    with open(args.metadata_out_file, 'w') as out_file:
        json.dump(metadata, out_file)

