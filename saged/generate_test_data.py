""" This file generates a dataset to be used in testing """

import json
import pickle
import random
import os

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

NUM_SAMPLES = 200
NUM_GENES = 5

current_dir = os.path.dirname(os.path.abspath(__file__))

compendium_path = os.path.join(current_dir, '../data/HOMO_SAPIENS.tsv')

# Pull 200 random sample names from the compendium
compendium_head = None
with open(compendium_path, 'r') as compendium_file:
    compendium_head = compendium_file.readline().strip().split('\t')

samples = random.sample(compendium_head, NUM_SAMPLES)
fake_expression = np.random.randn(NUM_GENES, NUM_SAMPLES)

# Assign each sample to a fake file
study_list = ['study1', 'study2', 'study3', 'study4', 'study5', 'study6']
study_probs = [.5, .2, .1, .1, .05, .05]

study_assignments = random.choices(study_list, weights=study_probs, k=NUM_SAMPLES)

# Generate formatted metadata mapping studies to samples
fake_metadata = {'experiments': {study: {'sample_accession_codes': []} for study in study_list}}
for sample, study in zip(samples, study_assignments):
    fake_metadata['experiments'][study]['sample_accession_codes'].append(sample)

# Write formatted metadata to a file
out_directory = os.path.join(current_dir, '../test/data')
metadata_file_path = os.path.join(out_directory, 'test_metadata.json')
with open(metadata_file_path, 'w') as metadata_file:
    json.dump(fake_metadata, metadata_file)

# Generate random labels for the samples
label_list = ['label1', 'label2', 'label3', 'label4', 'label5', 'label6']
label_probs = [.5, .2, .1, .1, .05, .05]

label_assignments = random.choices(label_list, weights=label_probs, k=NUM_SAMPLES)

# Format labels assignments in the same way as the main dataset
label_to_samples = {label: [] for label in label_list}
for label, sample in zip(label_assignments, samples):
    label_to_samples[label].append(sample)

sample_set = set(samples)
label_tuple = (label_to_samples, sample_set)

# Write label info to a file
label_file_path = os.path.join(out_directory, 'test_labels.pkl')
with open(label_file_path, 'wb') as out_file:
    pickle.dump(label_tuple, out_file)

# Create pandas dataframe from data
gene_names = []
for i in range(NUM_GENES):
    gene_base = 'ENSG0000000000'
    gene_name = '{}{}'.format(gene_base, i)
    gene_names.append(gene_name)

df = pd.DataFrame(data=fake_expression,
                  index=gene_names,
                  columns=samples
                  )

expression_out_path = os.path.join(out_directory, 'test_expression.tsv')
df.to_csv(expression_out_path, sep='\t')
