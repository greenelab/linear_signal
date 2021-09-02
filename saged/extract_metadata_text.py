"""
This script converts the metadata from recount_metadata.tsv into a form
usable by bioBERT
"""

import argparse

import pandas as pd
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata_file', help='The tsv containing recount3 metadata')
    parser.add_argument('out_file', help='The path to store the results to')
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_file, sep='\t')

    empty_count = 0
    with open(args.out_file, 'w') as out_file:
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            title = row['sra.sample_title']
            experiment_description = row['sra.design_description']
            sample_description = row['sra.sample_description']
            if str(title) == 'nan':
                title = ''
            if str(experiment_description) == 'nan':
                experiment_description = ''
            if str(sample_description) == 'nan':
                sample_description = ''

            metadata_string = ' '.join([str(title),
                                        str(experiment_description),
                                        str(sample_description)])

            metadata_string = metadata_string.strip()

            if len(metadata_string) == 0:
                metadata_string = 'none'
                empty_count += 1

            out_file.write('{}\n'.format(metadata_string))

    print('{} samples without metadata'.format(empty_count))
