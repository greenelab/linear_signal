"""
Convert gtex gene ids to genesymbols, normalize the expression, and write the resulting dataframe
to disk
"""

import argparse

import numpy as np
import pandas as pd


def parse_gtex_file(file_path: str) -> pd.DataFrame:
    """
    Read the expression data from a GTEx file

    Arguments
    ---------
    file_path: The path to the file containing GTEx expression data

    Returns
    -------
    expression_df: A samples x genes dataframe containing the expression data
    """
    with open(file_path) as in_file:
        # Throw away version string
        in_file.readline()
        gene_count, sample_count = in_file.readline().strip().split()
        expression_df = pd.read_csv(in_file, sep='\t', header=0)
        expression_df = expression_df.set_index('Name')
        expression_df = expression_df.drop('Description', axis='columns')
        expression_df = expression_df.T

        try:
            assert len(expression_df.columns) == int(gene_count)
            assert len(expression_df.index) == int(sample_count)
        except AssertionError:
            err = ('Expected {} rows and {} columns, '
                   'got {} and {}'.format(sample_count,
                                          gene_count,
                                          len(expression_df.index),
                                          len(expression_df.columns)))
            raise AssertionError(err)

    return expression_df


def standardize_column(col: pd.Series) -> pd.Series:
    """Zero-one standardize a dataframe column"""
    max_val = col.max()
    min_val = col.min()
    col_range = max_val - min_val

    if col_range == 0:
        standardized_column = np.zeros(len(col))
    else:
        standardized_column = (col - min_val) / col_range

    return standardized_column


def normalize_data(expression_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data so that each gene's entries will fall between zero and one

    Arguments
    ---------
    expression_df: A samples x genes dataframe containing the expression data

    Returns
    -------
    normalized_df: A df where each gene's values are scaled to between zero and one
    """
    normalized_df = expression_df.apply(standardize_column, axis=0)
    return normalized_df


def subset_data(expression_df: pd.DataFrame, n_genes) -> pd.DataFrame:
    """
    Subset data to the n most variable genes in the dataset

    Arguments
    ---------
    expression_df: A samples x genes dataframe containing the expression data

    Returns
    -------
    subset_df: The dataframe passed in with all but the most variable columns removed
    """
    per_gene_variances = expression_df.var()
    top_indices = np.argpartition(per_gene_variances, -n_genes)[-n_genes:]

    subset_df = expression_df.iloc[:, top_indices]

    return subset_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tpm_file', help='The expression data file downloaded from GTEx')
    parser.add_argument('out_file', help='The path to save the result to')
    parser.add_argument('--num_genes', help='The script will keep the top K most variable genes. '
                                            'This flag sets K.', default=5000, type=int)
    args = parser.parse_args()

    expression_df = parse_gtex_file(args.tpm_file)
    normalized_df = normalize_data(expression_df)
    subset_df = subset_data(normalized_df, args.num_genes)

    subset_df.to_csv(args.out_file, sep='\t')
