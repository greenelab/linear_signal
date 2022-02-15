"""
This script preprocesses TCGA data by dropping NaN genes, converting to TPM, standardizing,
and keeping the most variable genes
"""

# Drop Nan genes
# Convert to tpm
# Standardize


import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_expression(expression_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop genes with NaN values, convert FPKM to TPM, and zero-one standardize

    Arguments
    ---------
    expression_df - TCGA expression data stored in a in a genes x samples dataframe

    Returns
    -------
    preprocessed_df - The preprocessed expression data in samples x genes format
    """

    # Remove NaN genes
    expression_df = expression_df.loc[:, expression_df.isna().sum() == 0]

    # Convert to TPM by dividing each entry by the sum of the FPKM for that sample
    expression_df.div(expression_df.sum(), axis=1)

    # Standardize
    expression_df = expression_df.T
    scaler = MinMaxScaler()
    scaled_matrix = scaler.fit_transform(expression_df)
    preprocessed_df = pd.DataFrame(scaled_matrix, index=expression_df.index,
                                   columns=expression_df.columns)

    return preprocessed_df


def subset_genes(tpm_df: pd.DataFrame, n_genes: int) -> pd.DataFrame:
    """
    Subset data to the n most variable genes in the dataset

    Arguments
    ---------
    tpm_df: A samples x genes dataframe containing the preprocessed expression data
    n_genes: The number of genes to subset to
    Returns
    -------
    subset_df: The dataframe passed in with all but the most variable columns removed
    """
    per_gene_variances = tpm_df.var()
    top_indices = np.argpartition(per_gene_variances, -n_genes)[-n_genes:]

    subset_df = tpm_df.iloc[:, top_indices]

    return subset_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expression_file', help='The expression data file downloaded from TCGA')
    parser.add_argument('out_file', help='The path to save the result to')
    parser.add_argument('--num_genes', help='The script will keep the top K most variable genes. '
                                            'This flag sets K.', default=5000, type=int)
    args = parser.parse_args()

    expression_df = expression_df = pd.read_csv(args.expression_file, sep='\t', index_col=0)
    tpm_df = preprocess_expression(expression_df)
    subset_df = subset_genes(tpm_df, args.num_genes)

    subset_df.to_csv(args.out_file, sep='\t')
