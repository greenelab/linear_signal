# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: saged
#     language: python
#     name: saged
# ---

# %% [markdown]
# # Benchmark Results
# This notebook visualizes the output from the different models on different classification problems

# %%
import collections
import glob
import json
import os

import numpy as np
import pandas as pd
from plotnine import *

from saged import utils


# %% [markdown]
# ## Set Up Functions and Get Metadata

# %%
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


# %%
def return_unlabeled():
    # For use in a defaultdict
    return 'unlabeled'


# %%
data_dir = '../../data/'
map_file = os.path.join(data_dir, 'sample_classifications.pkl')

sample_to_label = utils.parse_map_file(map_file)
sample_to_label = collections.defaultdict(return_unlabeled, sample_to_label)

# %%
metadata_path = os.path.join(data_dir, 'aggregated_metadata.json')
metadata = None
with open(metadata_path) as json_file:
    metadata = json.load(json_file)
sample_metadata = metadata['samples']

# %%
experiments = metadata['experiments']
sample_to_study = {}
for study in experiments:
    for accession in experiments[study]['sample_accession_codes']:
        sample_to_study[accession] = study

# %% [markdown]
# ## Sepsis classification

# %%
in_files = glob.glob('../../results/single_label.*')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])

# %%
sepsis_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sepsis.')[-1]
    model_info = model_info.split('.')
        
    if len(model_info) == 4:
        unsupervised_model = model_info[0]
        supervised_model = model_info[1]
    else:
        unsupervised_model = 'untransformed'
        supervised_model = model_info[0]
             
    new_df['unsupervised'] = unsupervised_model
    new_df['supervised'] = supervised_model
        
    sepsis_metrics = pd.concat([sepsis_metrics, new_df])
    
sepsis_metrics

# %%
plot = ggplot(sepsis_metrics, aes(x='supervised', y='accuracy', fill='unsupervised')) 
plot += geom_violin()
plot += ggtitle('PCA vs untransformed data for classifying sepsis')
print(plot)

# %%
plot = ggplot(sepsis_metrics, aes(x='supervised', y='accuracy', fill='unsupervised')) 
plot += geom_jitter(size=3)
plot += ggtitle('PCA vs untransformed data for classifying sepsis')
print(plot)

# %% [markdown]
# ## All labels

# %%
in_files = glob.glob('../../results/all_labels.*')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])

# %%
metrics = None
for path in in_files:
    if metrics is None:
        metrics = pd.read_csv(path, sep='\t')
        
        model_info = path.strip('.tsv').split('all_labels.')[-1]
        
        model_info = model_info.split('.')
        
        if len(model_info) == 4:
            unsupervised_model = model_info[0]
            supervised_model = model_info[1]
        else:
            unsupervised_model = 'untransformed'
            supervised_model = model_info[0]
             
        metrics['unsupervised'] = unsupervised_model
        metrics['supervised'] = supervised_model
    else:
        new_df = pd.read_csv(path, sep='\t')
        model_info = path.strip('.tsv').split('all_labels.')[-1]
        model_info = model_info.split('.')
        
        if len(model_info) == 4:
            unsupervised_model = model_info[0]
            supervised_model = model_info[1]
        else:
            unsupervised_model = 'untransformed'
            supervised_model = model_info[0]
             
        new_df['unsupervised'] = unsupervised_model
        new_df['supervised'] = supervised_model
        
        metrics = pd.concat([metrics, new_df])

metrics

# %%
plot = ggplot(metrics, aes(x='supervised', y='accuracy', fill='unsupervised')) 
plot += geom_violin()
plot += ggtitle('PCA vs untransformed data for all label classification')
print(plot)

# %%
plot = ggplot(metrics, aes(x='supervised', y='accuracy', fill='unsupervised')) 
plot += geom_jitter(size=2)
plot += ggtitle('PCA vs untransformed data for all label classification')
print(plot)

# %% [markdown]
# # Subsets of healthy labels

# %%
in_files = glob.glob('../../results/subset_label.sepsis*')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])

# %%
sepsis_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sepsis.')[-1]
    model_info = model_info.split('.')
        
    if len(model_info) == 4:
        unsupervised_model = model_info[0]
        supervised_model = model_info[1]
    else:
        unsupervised_model = 'untransformed'
        supervised_model = model_info[0]
             
    new_df['unsupervised'] = unsupervised_model
    new_df['supervised'] = supervised_model
        
    sepsis_metrics = pd.concat([sepsis_metrics, new_df])
    
sepsis_metrics = sepsis_metrics.rename({'fraction of healthy used': 'healthy_used'}, axis='columns')
sepsis_metrics['healthy_used'] = sepsis_metrics['healthy_used'].round(1)
    
sepsis_metrics

# %%
print(sepsis_metrics[sepsis_metrics['healthy_used'] == 1])

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', )) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='unsupervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %% [markdown]
# ## Same analysis, but with tb instead of sepsis

# %%
in_files = glob.glob('../../results/subset_label.tb*')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])

# %%
tuberculosis_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('tb.')[-1]
    model_info = model_info.split('.')
        
    if len(model_info) == 4:
        unsupervised_model = model_info[0]
        supervised_model = model_info[1]
    else:
        unsupervised_model = 'untransformed'
        supervised_model = model_info[0]
             
    new_df['unsupervised'] = unsupervised_model
    new_df['supervised'] = supervised_model
        
    tuberculosis_metrics = pd.concat([tuberculosis_metrics, new_df])
    
tuberculosis_metrics = tuberculosis_metrics.rename({'fraction of healthy used': 'healthy_used'}, axis='columns')
tuberculosis_metrics['healthy_used'] = tuberculosis_metrics['healthy_used'].round(1)
tuberculosis_metrics

# %%
print(tuberculosis_metrics[tuberculosis_metrics['healthy_used'] == 1])

# %%
plot = ggplot(tuberculosis_metrics, aes(x='factor(healthy_used)', y='accuracy')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
plot = ggplot(tuberculosis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='unsupervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
plot = ggplot(tuberculosis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %% [markdown]
# ## Supervised Results Only
# The results above show that unsupervised learning mostly hurts performance rather than helping.
# The visualizations below compare each model based only on its supervised results.

# %%
supervised_sepsis = sepsis_metrics[sepsis_metrics['unsupervised'] == 'untransformed']

# %%
plot = ggplot(supervised_sepsis, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
supervised_tb = tuberculosis_metrics[tuberculosis_metrics['unsupervised'] == 'untransformed']

# %%
plot = ggplot(supervised_tb, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
plot = ggplot(supervised_tb, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_boxplot()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %% [markdown]
# ## Batch Effect Correction

# %%
in_files = glob.glob('../../results/subset_label.sepsis*be_corrected.tsv')
print(in_files[:5])

# %%
sepsis_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sepsis.')[-1]
    print(model_info)
    model_info = model_info.split('.')
    print(model_info)
        
    supervised_model = model_info[0]
             
    new_df['supervised'] = supervised_model
        
    sepsis_metrics = pd.concat([sepsis_metrics, new_df])
    
sepsis_metrics = sepsis_metrics.rename({'fraction of healthy used': 'healthy_used'}, axis='columns')
sepsis_metrics['healthy_used'] = sepsis_metrics['healthy_used'].round(1)
    
sepsis_metrics

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', )) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %% [markdown]
# ## TB Batch effect corrected

# %%
in_files = glob.glob('../../results/subset_label.tb*be_corrected.tsv')
print(in_files[:5])

# %%
tuberculosis_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('tb.')[-1]
    model_info = model_info.split('.')
        
    supervised_model = model_info[0]
             
    new_df['supervised'] = supervised_model
        
    tuberculosis_metrics = pd.concat([tuberculosis_metrics, new_df])
    
tuberculosis_metrics = tuberculosis_metrics.rename({'fraction of healthy used': 'healthy_used'}, axis='columns')
tuberculosis_metrics['healthy_used'] = tuberculosis_metrics['healthy_used'].round(1)
tuberculosis_metrics

# %%
plot = ggplot(tuberculosis_metrics, aes(x='factor(healthy_used)', y='accuracy')) 
plot += geom_boxplot()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
plot = ggplot(tuberculosis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %% [markdown]
# ## Better Metrics, Same Label Distribution in Train and Val sets

# %%
in_files = glob.glob('../../results/keep_ratios.sepsis*be_corrected.tsv')
print(in_files[:5])

# %%
sepsis_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sepsis.')[-1]
    model_info = model_info.split('.')
        
    supervised_model = model_info[0]
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-2]
        
    sepsis_metrics = pd.concat([sepsis_metrics, new_df])
    
sepsis_metrics = sepsis_metrics.rename({'fraction of data used': 'healthy_used'}, axis='columns')
sepsis_metrics['healthy_used'] = sepsis_metrics['healthy_used'].round(1)

# Looking at the training curves, deep_net isn't actually training
# I need to fix it going forward, but for now I can clean up the visualizations by removing it
sepsis_metrics = sepsis_metrics[~(sepsis_metrics['supervised'] == 'deep_net')]
sepsis_metrics['supervised'] = sepsis_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
sepsis_metrics

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy')) 
plot += geom_boxplot()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='balanced_accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
sepsis_stat_df = create_dataset_stat_df(sepsis_metrics, 
                                        sample_to_study, 
                                        sample_metadata, 
                                        sample_to_label,
                                        'sepsis')

sepsis_stat_df.tail(5)

# %%
ggplot(sepsis_stat_df, aes(x='train_val_diff', 
                           y='balanced_accuracy', 
                           color='val_disease_count')) + geom_point() + facet_grid('model ~ .')

# %%
plot = ggplot(sepsis_metrics, aes(x='train sample count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Effect of All Sepsis Data')
plot

# %% [markdown]
# ## Same Distribution Tuberculosis

# %%
in_files = glob.glob('../../results/keep_ratios.tb*be_corrected.tsv')
print(in_files[:5])

# %%
tb_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('tb.')[-1]
    model_info = model_info.split('.')
        
    supervised_model = model_info[0]
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-2]
        
    tb_metrics = pd.concat([tb_metrics, new_df])
    
tb_metrics = tb_metrics.rename({'fraction of data used': 'healthy_used'}, axis='columns')
tb_metrics['healthy_used'] = tb_metrics['healthy_used'].round(1)
tb_metrics

# %%
plot = ggplot(tb_metrics, aes(x='factor(healthy_used)', y='accuracy')) 
plot += geom_boxplot()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
plot = ggplot(tb_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
plot = ggplot(tb_metrics, aes(x='factor(healthy_used)', y='balanced_accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)

# %%
tb_stat_df = create_dataset_stat_df(tb_metrics, 
                                    sample_to_study, 
                                    sample_metadata, 
                                    sample_to_label,
                                    'tb')

tb_stat_df.tail(5)

# %%
ggplot(tb_stat_df, aes(x='train_val_diff', 
                       y='balanced_accuracy', 
                       color='val_disease_count')) + geom_point() + facet_grid('model ~ .')

# %%
plot = ggplot(tb_metrics, aes(x='train sample count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot

# %% [markdown]
# ## Results from Small Datasets

# %%
in_files = glob.glob('../../results/small_subsets.sepsis*be_corrected.tsv')
print(in_files[:5])

# %%
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

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(train_count)', y='balanced_accuracy')) 
plot += geom_boxplot()
plot += ggtitle('Sepsis Dataset Size Effects (equal label counts)')
print(plot)

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(train_count)', y='balanced_accuracy', fill='supervised')) 
plot += geom_boxplot()
plot += ggtitle('Sepsis Datset Size by Model (equal label counts)')
print(plot)

# %%
plot = ggplot(sepsis_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Sepsis Crossover Point')
plot

# %% [markdown]
# ## Small Training Set TB

# %%
in_files = glob.glob('../../results/small_subsets.tb*be_corrected.tsv')
print(in_files[:5])

# %%
tb_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('tb.')[-1]
    model_info = model_info.split('.')
        
    supervised_model = model_info[0]
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-2]
        
    tb_metrics = pd.concat([tb_metrics, new_df])
    
tb_metrics['train_count'] = tb_metrics['train sample count']

# Looking at the training curves, deep_net isn't actually training
# I need to fix it going forward, but for now I can clean up the visualizations by removing it
tb_metrics = tb_metrics[~(tb_metrics['supervised'] == 'deep_net')]
tb_metrics['supervised'] = tb_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
tb_metrics

# %%
plot = ggplot(tb_metrics, aes(x='factor(train_count)', y='balanced_accuracy')) 
plot += geom_boxplot()
plot += ggtitle('TB Dataset Size Effects (equal label counts)')
print(plot)

# %%
plot = ggplot(tb_metrics, aes(x='factor(train_count)', y='balanced_accuracy', fill='supervised')) 
plot += geom_boxplot()
plot += ggtitle('TB Dataset Size vs Models (equal label counts)')
print(plot)

# %%
plot = ggplot(tb_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth(method='loess')
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('TB (lack of a) Crossover Point')
plot

# %% [markdown]
# ## Small training sets without be correction

# %%
in_files = glob.glob('../../results/small_subsets.sepsis*.tsv')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])

# %%
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

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(train_count)', y='balanced_accuracy')) 
plot += geom_boxplot()
plot += ggtitle('Sepsis Dataset Size Effects (equal label counts)')
print(plot)

# %%
plot = ggplot(sepsis_metrics, aes(x='factor(train_count)', y='balanced_accuracy', fill='supervised')) 
plot += geom_boxplot()
plot += ggtitle('Sepsis Datset Size by Model (equal label counts)')
print(plot)

# %%
plot = ggplot(sepsis_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Sepsis Crossover Point')
plot

# %% [markdown]
# ## TB Not Batch Effect Corrected

# %%
in_files = glob.glob('../../results/small_subsets.tb*.tsv')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])

# %%
tb_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('tb.')[-1]
    model_info = model_info.split('.')
        
    supervised_model = model_info[0]
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-2]
        
    tb_metrics = pd.concat([tb_metrics, new_df])
    
tb_metrics['train_count'] = tb_metrics['train sample count']

# Looking at the training curves, deep_net isn't actually training
# I need to fix it going forward, but for now I can clean up the visualizations by removing it
tb_metrics = tb_metrics[~(tb_metrics['supervised'] == 'deep_net')]
tb_metrics['supervised'] = tb_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
tb_metrics

# %%
plot = ggplot(tb_metrics, aes(x='factor(train_count)', y='balanced_accuracy')) 
plot += geom_boxplot()
plot += ggtitle('tb Dataset Size Effects (equal label counts)')
print(plot)

# %%
plot = ggplot(tb_metrics, aes(x='factor(train_count)', y='balanced_accuracy', fill='supervised')) 
plot += geom_boxplot()
plot += ggtitle('tb Datset Size by Model (equal label counts)')
print(plot)

# %%
plot = ggplot(tb_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('tb Crossover Point')
plot

# %% [markdown]
# ## Large training sets without be correction

# %%
in_files = glob.glob('../../results/keep_ratios.sepsis*.tsv')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])

# %%
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

# %%
plot = ggplot(sepsis_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Sepsis Crossover Point')
plot

# %% [markdown]
# ## TB Not Batch Effect Corrected

# %%
in_files = glob.glob('../../results/keep_ratios.tb*.tsv')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])

# %%
tb_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('tb.')[-1]
    model_info = model_info.split('.')
        
    supervised_model = model_info[0]
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-2]
        
    tb_metrics = pd.concat([tb_metrics, new_df])
    
tb_metrics['train_count'] = tb_metrics['train sample count']

# Looking at the training curves, deep_net isn't actually training
# I need to fix it going forward, but for now I can clean up the visualizations by removing it
tb_metrics = tb_metrics[~(tb_metrics['supervised'] == 'deep_net')]
tb_metrics['supervised'] = tb_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
tb_metrics

# %%
plot = ggplot(tb_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('tb Crossover Point')
plot
