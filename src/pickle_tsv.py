import argparse
import gc

import numpy as np
import pandas as pd
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tsv_path',
                        help='The path to the tsv to turn into a dataframe and pickle',
                        )
    parser.add_argument('out_path',
                        help='The path to save the results to'
                        )
    args = parser.parse_args()

    dtypes = {'study': str}

    with open(args.tsv_path) as tsv_file:
        header = tsv_file.readline()
        header = header.strip().split('\t')[1:]

        for col in header:
            dtypes[col] = np.half  # half precision floats

    chunksize = 10000

    df = pd.DataFrame()
    data_iterator = pd.read_csv(args.tsv_path, sep='\t', index_col=0, dtype=dtypes,
                                chunksize=chunksize)
    for sub_data in tqdm.tqdm(data_iterator, total=32):
        df = df.append(sub_data)
        gc.collect()

    df.to_pickle(args.out_path)
