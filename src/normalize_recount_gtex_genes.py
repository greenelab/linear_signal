"""
This script normalizes the data from recount3 and keeps the same set of genes
that were used to run GTEx.

This script is similar to normalize_recount, except that we go into it with a set
of genes we know we want to keep
"""

import argparse
from typing import Dict

import numpy as np
import pandas as pd
import tqdm

import utils


def parse_gene_lengths(file_path: str) -> Dict[str, int]:
    """Parses a tsv file containing genes and their length

    Arguments
    ---------
    file_path - The path to the file mapping genes to lengths

    Returns
    -------
    gene_to_len - A dict mapping ensembl gene ids to their length in base pairs
    """
    gene_to_len = {}
    with open(file_path) as in_file:
        # Throw out header
        in_file.readline()
        for line in in_file:
            line = line.replace('"', '')
            gene, length = line.strip().split('\t')
            try:
                gene_to_len[gene] = int(length)
            except ValueError:
                # Some genes have no length, but will be removed in a later step
                pass
    return gene_to_len


def calculate_tpm(counts: np.ndarray, gene_length_arr: np.ndarray) -> np.ndarray:
    """"Given an array of counts, calculate the transcripts per kilobase million
    based on the steps here:
    https://www.rna-seqblog.com/rpkm-fpkm-and-tpm-clearly-explained/

    Arguments
    ---------
    counts: The array of transcript counts per gene
    gene_length_arr: The array of lengths for each gene in counts

    Returns
    -------
    tpm: The tpm normalized expression data
    """
    counts = np.array(counts, dtype=float)

    reads_per_kb = counts / gene_length_arr

    sample_total_counts = np.sum(reads_per_kb)
    per_million_transcripts = sample_total_counts / 1e6

    tpm = reads_per_kb / per_million_transcripts

    return tpm


# Fill in holes in gene lengths mapping
NEW_LENGTHS = {'ENSG00000264608': 458,
               'ENSG00000241043': 2382,
               'ENSG00000272301': 481,
               'ENSG00000275142': 1633,
               'ENSG00000233895': 1305,
               'ENSG00000258297': 2339,
               'ENSG00000241978': 6866,
               'ENSG00000256374': 760,
               'ENSG00000273478': 657,
               'ENSG00000268568': 1328,
               'ENSG00000268568': 1328,
               'ENSG00000130723': 266,
               'ENSG00000256248': 192,
               'ENSG00000274897': 1675,
               'ENSG00000284413': 5999,
               'ENSG00000213865': 575,
               'ENSG00000277420': 2676,
               'ENSG00000228439': 1873,
               'ENSG00000256164': 2779,
               'ENSG00000242687': 12675,
               'ENSG00000270015': 4849,
               'ENSG00000116957': 5374}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('count_file', help='The file containing the count matrix generated by '
                                           'download_recount3.R')
    parser.add_argument('gene_file', help='The file with gene lengths from get_gene_lengths.R')
    parser.add_argument('out_file', help='The file to save the normalized results to')
    parser.add_argument('metadata_file', help='The file with info mapping samples to studies')
    parser.add_argument('gtex_file', help='File containing GTEx training data ')

    # The number of lines in the count file for calculating the number of times
    # the loops will run. If this is incorrect the script's output will
    # still be correct, but the progress bar will be slightly off
    LINES_IN_FILE = 317259

    args = parser.parse_args()

    sample_to_study = utils.recount_map_sample_to_study(args.metadata_file)
    gene_to_len = parse_gene_lengths(args.gene_file)
    gene_to_len.update(NEW_LENGTHS)

    with open(args.count_file, 'r') as count_file:
        header = count_file.readline()
        header = header.replace('"', '')
        header_genes = header.strip().split('\t')
        header_genes = [gene.split('.')[0] for gene in header_genes]

        bad_indices = []
        gene_length_arr = []
        for i, gene in enumerate(header_genes):
            if gene not in gene_to_len.keys():
                bad_indices.append(i)
            else:
                gene_length_arr.append(gene_to_len[gene])

        gene_length_arr = np.array(gene_length_arr)

        means = None
        M2 = None
        maximums = None
        minimums = None

        samples_seen = set()

        # First time through the data, calculate statistics
        for i, line in tqdm.tqdm(enumerate(count_file), total=LINES_IN_FILE):
            line = line.replace('"', '')
            line = line.strip().split('\t')
            sample = line[0]

            # Remove duplicates
            if sample in samples_seen:
                continue
            samples_seen.add(sample)

            # https://github.com/LieberInstitute/recount3/issues/5
            if sample not in sample_to_study:
                print('Skipping {}'.format(sample))
                continue
            try:
                # Thanks to stackoverflow for this smart optimization
                # https://stackoverflow.com/a/11303234/10930590
                counts = line[1:]  # bad_indices is still correct because of how R saves tables
                for index in reversed(bad_indices):
                    del counts[index]

                tpm = calculate_tpm(counts, gene_length_arr)

                if any(np.isnan(tpm)):
                    continue

                # Online variance calculation https://stackoverflow.com/a/15638726/10930590
                if means is None:
                    means = tpm
                    M2 = 0
                    maximums = tpm
                    minimums = tpm
                else:
                    delta = tpm - means
                    means = means + delta / (i + 1)
                    M2 = M2 + delta * (tpm - means)
                    maximums = np.maximum(maximums, tpm)
                    minimums = np.minimum(minimums, tpm)

            except ValueError as e:
                # Throw out malformed lines caused by issues with downloading data
                print(e)

        per_gene_variances = M2 / (i-1)
        max_min_diff = maximums - minimums

        out_file = open(args.out_file, 'w')

        header = header.strip().split('\t')

        gtex_data = pd.read_csv(args.gtex_file, sep='\t', index_col=0)
        genes_to_keep = list(gtex_data.columns)

        for gene in genes_to_keep:
            gene = gene.split('.')[0]
            assert gene in gene_to_len

        keep_indices = np.in1d(header, genes_to_keep)

        # Use numpy to allow indexing with a list of indices
        header_arr = np.array(header)
        header_arr = header_arr[keep_indices]

        keep_indices = np.delete(keep_indices, bad_indices)
        assert sum(keep_indices) == 5000

        header = header_arr.tolist()

        for item in header:
            assert item in genes_to_keep

        header = 'sample\t' + '\t'.join(header)
        out_file.write(header)
        out_file.write('\n')

    with open(args.count_file, 'r') as count_file:
        samples_seen = set()
        # Second time through the data - standardize and write outputs
        for i, line in tqdm.tqdm(enumerate(count_file), total=LINES_IN_FILE):
            line = line.replace('"', '')
            line = line.strip().split('\t')
            sample = line[0]

            if sample in samples_seen:
                continue
            samples_seen.add(sample)

            if sample not in sample_to_study:
                continue
            try:
                counts = line[1:]  # bad_indices is still correct because of how R saves tables
                for index in reversed(bad_indices):
                    del counts[index]

                tpm = calculate_tpm(counts, gene_length_arr)

                if any(np.isnan(tpm)):
                    continue

                # Zero-one standardize
                standardized_tpm = (tpm - minimums) / max_min_diff

                # Keep only most variable genes
                most_variable_tpm = standardized_tpm[keep_indices]
                tpm_list = most_variable_tpm.tolist()
                tpm_strings = ['{}'.format(x) for x in tpm_list]

                out_file.write('{}\t'.format(sample))
                out_file.write('\t'.join(tpm_strings))
                out_file.write('\n')

            except ValueError as e:
                # Throw out malformed lines caused by issues with downloading data
                print(e)
