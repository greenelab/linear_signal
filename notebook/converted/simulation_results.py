#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import glob
import json
import os

import numpy as np
import pandas as pd
from plotnine import *

from saged import utils


# ## Set Up Functions and Get Metadata

# In[2]:


def split_sample_names(df_row):
    train_samples = df_row['train samples'].split(',')
    val_samples = df_row['val samples'].split(',')
    
    return train_samples, val_samples

def create_dataset_stat_df(metrics_df, sample_to_study, 
                           sample_metadata, sample_to_label, disease,):

    data_dict = {'train_disease_count': [],
                 'train_healthy_count': [],
                 'val_disease_count': [],
                 'val_healthy_count': [],
                 'accuracy': [],
                 'balanced_accuracy': [],
                 'subset_fraction': [],
                 'seed': [],
                 'model': []
                }
    for _, row in metrics_df.iterrows():
        # Keep analysis simple for now



        data_dict['seed'].append(row['seed'])    
        data_dict['subset_fraction'].append(row['healthy_used'])
        data_dict['accuracy'].append(row['accuracy'])
        data_dict['model'].append(row['supervised'])
        if 'balanced_accuracy' in row:
            data_dict['balanced_accuracy'].append(row['balanced_accuracy'])

        train_samples, val_samples = split_sample_names(row)

        (train_studies, train_platforms, 
        train_diseases, train_disease_counts) = get_dataset_stats(train_samples,
                                                                  sample_to_study,
                                                                  sample_metadata,
                                                                  sample_to_label)
        data_dict['train_disease_count'].append(train_diseases[disease])
        data_dict['train_healthy_count'].append(train_diseases['healthy'])


        (val_studies, val_platforms, 
        val_diseases, val_disease_counts) = get_dataset_stats(val_samples,
                                                              sample_to_study,
                                                              sample_metadata,
                                                              sample_to_label)
        data_dict['val_disease_count'].append(val_diseases[disease])
        data_dict['val_healthy_count'].append(val_diseases['healthy'])

    stat_df = pd.DataFrame.from_dict(data_dict)
    
    stat_df['train_disease_percent'] = (stat_df['train_disease_count'] / 
                                        (stat_df['train_disease_count'] + 
                                         stat_df['train_healthy_count']))
    
    stat_df['val_disease_percent'] = (stat_df['val_disease_count'] /
                                      (stat_df['val_disease_count'] + 
                                       stat_df['val_healthy_count']))
    
    stat_df['train_val_diff'] = abs(stat_df['train_disease_percent'] - 
                                    stat_df['val_disease_percent'])
    stat_df['train_count'] = (stat_df['train_disease_count'] + 
                              stat_df['train_healthy_count'])
    
    return stat_df

def get_dataset_stats(sample_list, sample_to_study, sample_metadata, sample_to_label):
    studies = []
    platforms = []
    diseases = []
    study_disease_counts = {}

    for sample in sample_list:
        study = sample_to_study[sample]
        studies.append(study)
        platform = sample_metadata[sample]['refinebio_platform'].lower()
        platforms.append(platform)

        disease = sample_to_label[sample]
        diseases.append(disease)
        
        if study in study_disease_counts:
            study_disease_counts[study][disease] = study_disease_counts[study].get(disease, 0) + 1
        else:
            study_disease_counts[study] = {disease: 1}
            
    studies = collections.Counter(studies)
    platforms = collections.Counter(platforms)
    diseases = collections.Counter(diseases)

    
    return studies, platforms, diseases, study_disease_counts


# In[3]:


def return_unlabeled():
    # For use in a defaultdict
    return 'unlabeled'


# In[4]:


data_dir = '../../data/simulated'
map_file = os.path.join(data_dir, 'simulation_labels.pkl')

sample_to_label = utils.parse_map_file(map_file)
sample_to_label = collections.defaultdict(return_unlabeled, sample_to_label)


# In[5]:


metadata_path = os.path.join(data_dir, 'sample_metadata.json')
metadata = None
with open(metadata_path) as json_file:
    metadata = json.load(json_file)
sample_metadata = metadata['samples']


# In[6]:


experiments = metadata['experiments']
sample_to_study = {}
for study in experiments:
    for accession in experiments[study]['sample_accession_codes']:
        sample_to_study[accession] = study


# ## Sepsis classification

# In[7]:


in_files = glob.glob('../../results/simulation.sepsis*.tsv')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])


# In[8]:


sepsis_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sepsis.')[-1]
    model_info = model_info.split('.')
        
    supervised_model = model_info[0]
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-2]
        
    sepsis_metrics = pd.concat([sepsis_metrics, new_df])
    
sepsis_metrics['train_count'] = sepsis_metrics['train sample count']

# Looking at the training curves, deep_net isn't actually training
# I need to fix it going forward, but for now I can clean up the visualizations by removing it
sepsis_metrics = sepsis_metrics[~(sepsis_metrics['supervised'] == 'deep_net')]
sepsis_metrics['supervised'] = sepsis_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
sepsis_metrics


# In[9]:


plot = ggplot(sepsis_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Simulated Sepsis Accuracy vs Sample Count')
plot


# In[12]:


plot = ggplot(sepsis_metrics[sepsis_metrics['supervised'] == 'three_layer_net'], aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.4)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Simulated Sepsis Accuracy vs Sample Count')
plot


# In[15]:


plot = ggplot(sepsis_metrics[sepsis_metrics['supervised'] == 'three_layer_net'], aes(x='balanced_accuracy'')) 
plot += geom_density()
plot += ggtitle('Simulated Sepsis Accuracy vs Sample Count')
plot


# In[ ]:




