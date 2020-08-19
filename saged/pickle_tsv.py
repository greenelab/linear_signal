import argparse

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tsv_path',
                        help='The path to the tsv to turn into a dataframe and pickle',
                        )
    parser.add_argument('out_path',
                        help='The path to save the results to'
                        )
    args = parser.parse_args()

    with open(args.tsv_path) as tsv_file:
        df = pd.read_csv(tsv_file, sep='\t', index_col=0)

    df.to_pickle(args.out_path)
