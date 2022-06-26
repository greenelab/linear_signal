"""
Create a label file and a pickled subset file of the recount3 dataset containing only samples
with manually annotated tissue labels
"""

import argparse
import pickle
from typing import Dict

import pandas as pd


def map_samples_to_labels(file_path: str) -> Dict[str, str]:
    metadata_df = pd.read_csv(file_path, delimiter='\t')
    columns_to_keep = ['external_id', 'recount_pred.curated.tissue']

    samples_and_labels = metadata_df.drop(metadata_df.columns.difference(columns_to_keep), 1)
    samples_and_labels = samples_and_labels.dropna(axis='rows')
    samples_and_labels = samples_and_labels.set_index('external_id')

    sample_to_label = samples_and_labels.to_dict()['recount_pred.curated.tissue']

    return sample_to_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('count_file', help='The pickled file containing the tpm matrix generated '
                                           'by pickle_tsv.py')
    parser.add_argument('metadata_file', help='The file with info mapping samples to studies')
    parser.add_argument('subset_out', help='The path to save the labeled data to')
    parser.add_argument('label_out', help='The path to save the sample to label mappings to')

    args = parser.parse_args()

    sample_to_label = map_samples_to_labels(args.metadata_file)

    with open(args.label_out, 'wb') as out_file:
        pickle.dump(sample_to_label, out_file)

    expression_df = None
    with open(args.count_file, 'rb') as count_file:
        expression_df = pickle.load(count_file)

    samples_to_keep = list(sample_to_label.keys())

    samples_in_tpm = set(expression_df.index)

    samples_to_keep = [s for s in samples_to_keep if s in samples_in_tpm]

    expression_df = expression_df.loc[samples_to_keep, :]

    with open(args.subset_out, 'wb') as out_file:
        pickle.dump(expression_df, out_file)
