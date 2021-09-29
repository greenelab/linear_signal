#!/usr/bin/env python
# coding: utf-8

# # Benchmark Results
# This notebook visualizes the output from the different models on different classification problems

# In[1]:


import collections
import glob
import json
import os

import numpy as np
import pandas as pd
from plotnine import *

from saged.utils import split_sample_names, create_dataset_stat_df, get_dataset_stats, parse_map_file


# ## Set Up Functions and Get Metadata

# In[3]:


def return_unlabeled():
    # For use in a defaultdict
    return 'unlabeled'


# In[4]:


data_dir = '../../data/'
map_file = os.path.join(data_dir, 'sample_classifications.pkl')

sample_to_label = parse_map_file(map_file)
sample_to_label = collections.defaultdict(return_unlabeled, sample_to_label)


# In[ ]:


metadata_path = os.path.join(data_dir, 'aggregated_metadata.json')
metadata = None
with open(metadata_path) as json_file:
    metadata = json.load(json_file)
sample_metadata = metadata['samples']


# In[ ]:


experiments = metadata['experiments']
sample_to_study = {}
for study in experiments:
    for accession in experiments[study]['sample_accession_codes']:
        sample_to_study[accession] = study


# ## Sepsis classification

# In[8]:


in_files = glob.glob('../../results/single_label.*')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])


# In[9]:


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


# In[10]:


plot = ggplot(sepsis_metrics, aes(x='supervised', y='accuracy', fill='unsupervised')) 
plot += geom_violin()
plot += ggtitle('PCA vs untransformed data for classifying sepsis')
print(plot)


# In[11]:


plot = ggplot(sepsis_metrics, aes(x='supervised', y='accuracy', fill='unsupervised')) 
plot += geom_jitter(size=3)
plot += ggtitle('PCA vs untransformed data for classifying sepsis')
print(plot)


# ## All labels

# In[12]:


in_files = glob.glob('../../results/all_labels.*')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])


# In[13]:


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


# In[14]:


plot = ggplot(metrics, aes(x='supervised', y='accuracy', fill='unsupervised')) 
plot += geom_violin()
plot += ggtitle('PCA vs untransformed data for all label classification')
print(plot)


# In[15]:


plot = ggplot(metrics, aes(x='supervised', y='accuracy', fill='unsupervised')) 
plot += geom_jitter(size=2)
plot += ggtitle('PCA vs untransformed data for all label classification')
print(plot)


# # Subsets of healthy labels

# In[16]:


in_files = glob.glob('../../results/subset_label.sepsis*')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])


# In[17]:


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


# In[18]:


print(sepsis_metrics[sepsis_metrics['healthy_used'] == 1])


# In[19]:


plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', )) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[20]:


plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='unsupervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[21]:


plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# ## Same analysis, but with tb instead of sepsis

# In[22]:


in_files = glob.glob('../../results/subset_label.tb*')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])


# In[23]:


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


# In[24]:


print(tuberculosis_metrics[tuberculosis_metrics['healthy_used'] == 1])


# In[25]:


plot = ggplot(tuberculosis_metrics, aes(x='factor(healthy_used)', y='accuracy')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[26]:


plot = ggplot(tuberculosis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='unsupervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[27]:


plot = ggplot(tuberculosis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# ## Supervised Results Only
# The results above show that unsupervised learning mostly hurts performance rather than helping.
# The visualizations below compare each model based only on its supervised results.

# In[28]:


supervised_sepsis = sepsis_metrics[sepsis_metrics['unsupervised'] == 'untransformed']


# In[29]:


plot = ggplot(supervised_sepsis, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[30]:


supervised_tb = tuberculosis_metrics[tuberculosis_metrics['unsupervised'] == 'untransformed']


# In[31]:


plot = ggplot(supervised_tb, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[32]:


plot = ggplot(supervised_tb, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_boxplot()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# ## Batch Effect Correction

# In[33]:


in_files = glob.glob('../../results/subset_label.sepsis*be_corrected.tsv')
print(in_files[:5])


# In[34]:


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


# In[35]:


plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', )) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[36]:


plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# ## TB Batch effect corrected

# In[37]:


in_files = glob.glob('../../results/subset_label.tb*be_corrected.tsv')
print(in_files[:5])


# In[38]:


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


# In[39]:


plot = ggplot(tuberculosis_metrics, aes(x='factor(healthy_used)', y='accuracy')) 
plot += geom_boxplot()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[40]:


plot = ggplot(tuberculosis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# ## Better Metrics, Same Label Distribution in Train and Val sets

# In[11]:


in_files = glob.glob('../../results/keep_ratios.sepsis*be_corrected.tsv')
print(in_files[:5])


# In[12]:


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


# In[13]:


plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy')) 
plot += geom_boxplot()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[14]:


plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[15]:


plot = ggplot(sepsis_metrics, aes(x='factor(healthy_used)', y='balanced_accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[16]:


sepsis_stat_df = create_dataset_stat_df(sepsis_metrics, 
                                        sample_to_study, 
                                        sample_metadata, 
                                        sample_to_label,
                                        'sepsis')

sepsis_stat_df.tail(5)


# In[17]:


ggplot(sepsis_stat_df, aes(x='train_val_diff', 
                           y='balanced_accuracy', 
                           color='val_disease_count')) + geom_point() + facet_grid('model ~ .')


# In[18]:


plot = ggplot(sepsis_metrics, aes(x='train sample count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Effect of All Sepsis Data')
plot


# ## Same Distribution Tuberculosis

# In[19]:


in_files = glob.glob('../../results/keep_ratios.tb*be_corrected.tsv')
print(in_files[:5])


# In[20]:


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


# In[21]:


plot = ggplot(tb_metrics, aes(x='factor(healthy_used)', y='accuracy')) 
plot += geom_boxplot()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[22]:


plot = ggplot(tb_metrics, aes(x='factor(healthy_used)', y='accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[23]:


plot = ggplot(tb_metrics, aes(x='factor(healthy_used)', y='balanced_accuracy', fill='supervised')) 
plot += geom_violin()
plot += ggtitle('Effect of subsetting healthy data on prediction accuracy')
print(plot)


# In[24]:


tb_stat_df = create_dataset_stat_df(tb_metrics, 
                                    sample_to_study, 
                                    sample_metadata, 
                                    sample_to_label,
                                    'tb')

tb_stat_df.tail(5)


# In[55]:


ggplot(tb_stat_df, aes(x='train_val_diff', 
                       y='balanced_accuracy', 
                       color='val_disease_count')) + geom_point() + facet_grid('model ~ .')


# In[25]:


plot = ggplot(tb_metrics, aes(x='train sample count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot


# ## Results from Small Datasets

# In[57]:


in_files = glob.glob('../../results/small_subsets.sepsis*be_corrected.tsv')
print(in_files[:5])


# In[58]:


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


# In[59]:


plot = ggplot(sepsis_metrics, aes(x='factor(train_count)', y='balanced_accuracy')) 
plot += geom_boxplot()
plot += ggtitle('Sepsis Dataset Size Effects (equal label counts)')
print(plot)


# In[60]:


plot = ggplot(sepsis_metrics, aes(x='factor(train_count)', y='balanced_accuracy', fill='supervised')) 
plot += geom_boxplot()
plot += ggtitle('Sepsis Datset Size by Model (equal label counts)')
print(plot)


# In[61]:


plot = ggplot(sepsis_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Sepsis Crossover Point')
plot


# ## Small Training Set TB

# In[62]:


in_files = glob.glob('../../results/small_subsets.tb*be_corrected.tsv')
print(in_files[:5])


# In[63]:


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


# In[64]:


plot = ggplot(tb_metrics, aes(x='factor(train_count)', y='balanced_accuracy')) 
plot += geom_boxplot()
plot += ggtitle('TB Dataset Size Effects (equal label counts)')
print(plot)


# In[65]:


plot = ggplot(tb_metrics, aes(x='factor(train_count)', y='balanced_accuracy', fill='supervised')) 
plot += geom_boxplot()
plot += ggtitle('TB Dataset Size vs Models (equal label counts)')
print(plot)


# In[66]:


plot = ggplot(tb_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth(method='loess')
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('TB (lack of a) Crossover Point')
plot


# ## Small training sets without be correction

# In[67]:


in_files = glob.glob('../../results/small_subsets.sepsis*.tsv')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])


# In[68]:


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


# In[69]:


plot = ggplot(sepsis_metrics, aes(x='factor(train_count)', y='balanced_accuracy')) 
plot += geom_boxplot()
plot += ggtitle('Sepsis Dataset Size Effects (equal label counts)')
print(plot)


# In[70]:


plot = ggplot(sepsis_metrics, aes(x='factor(train_count)', y='balanced_accuracy', fill='supervised')) 
plot += geom_boxplot()
plot += ggtitle('Sepsis Datset Size by Model (equal label counts)')
print(plot)


# In[71]:


plot = ggplot(sepsis_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Sepsis Crossover Point')
plot


# ## TB Not Batch Effect Corrected

# In[72]:


in_files = glob.glob('../../results/small_subsets.tb*.tsv')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])


# In[73]:


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


# In[74]:


plot = ggplot(tb_metrics, aes(x='factor(train_count)', y='balanced_accuracy')) 
plot += geom_boxplot()
plot += ggtitle('tb Dataset Size Effects (equal label counts)')
print(plot)


# In[75]:


plot = ggplot(tb_metrics, aes(x='factor(train_count)', y='balanced_accuracy', fill='supervised')) 
plot += geom_boxplot()
plot += ggtitle('tb Datset Size by Model (equal label counts)')
print(plot)


# In[76]:


plot = ggplot(tb_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('tb Crossover Point')
plot


# ## Large training sets without be correction

# In[6]:


in_files = glob.glob('../../results/keep_ratios.sepsis*.tsv')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])


# In[9]:


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

sepsis_metrics['supervised'] = sepsis_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
sepsis_metrics


# In[10]:


plot = ggplot(sepsis_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Sepsis Crossover Point')
plot


# ## TB Not Batch Effect Corrected

# In[80]:


in_files = glob.glob('../../results/keep_ratios.tb*.tsv')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])


# In[81]:


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


# In[82]:


plot = ggplot(tb_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('tb Crossover Point')
plot


# ## Lupus Analyses

# In[83]:


in_files = glob.glob('../../results/keep_ratios.lupus*.tsv')
in_files = [file for file in in_files if 'be_corrected' in file]
print(in_files[:5])


# In[84]:


lupus_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('lupus.')[-1]
    model_info = model_info.split('.')
        
    supervised_model = model_info[0]
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-2]
        
    lupus_metrics = pd.concat([lupus_metrics, new_df])
    
lupus_metrics['train_count'] = lupus_metrics['train sample count']

# Looking at the training curves, deep_net isn't actually training
# I need to fix it going forward, but for now I can clean up the visualizations by removing it
lupus_metrics = lupus_metrics[~(lupus_metrics['supervised'] == 'deep_net')]
lupus_metrics['supervised'] = lupus_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
lupus_metrics


# In[85]:


plot = ggplot(lupus_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('lupus Crossover Point')
plot


# ## Lupus Not Batch Effect Corrected

# In[86]:


in_files = glob.glob('../../results/keep_ratios.lupus*.tsv')
in_files = [file for file in in_files if 'be_corrected' not in file]
print(in_files[:5])


# In[87]:


lupus_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('lupus.')[-1]
    model_info = model_info.split('.')
        
    supervised_model = model_info[0]
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-2]
        
    lupus_metrics = pd.concat([lupus_metrics, new_df])
    
lupus_metrics['train_count'] = lupus_metrics['train sample count']

# Looking at the training curves, deep_net isn't actually training
# I need to fix it going forward, but for now I can clean up the visualizations by removing it
lupus_metrics = lupus_metrics[~(lupus_metrics['supervised'] == 'deep_net')]
lupus_metrics['supervised'] = lupus_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
lupus_metrics


# In[88]:


plot = ggplot(lupus_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('lupus Crossover Point')
plot


# ## Tissue Prediction

# In[15]:


in_files = glob.glob('../../results/Blood.Breast.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[16]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('Breast.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['supervised'] = tissue_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics


# In[17]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Blood vs Breast Tissue Prediction')
plot


# ### BE Corrected binary tissue classification

# In[5]:


in_files = glob.glob('../../results/Blood.Breast.*.tsv')
in_files = [f for f in in_files if 'be_corrected' in f]
print(in_files[:5])


# In[6]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('Breast.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['supervised'] = tissue_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics


# In[7]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Blood vs Breast Tissue Prediction')
plot


# ### All Tissue Predictions

# In[18]:


in_files = glob.glob('../../results/all-tissue.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[19]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('all-tissue.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['supervised'] = tissue_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['supervised'] = tissue_metrics['supervised'].str.replace('deep_net', 'five_layer_net')
tissue_metrics


# In[20]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Multiclass Tissue Prediction')
plot


# ## Imputation pretraining

# In[11]:


in_files = glob.glob('../../results/tissue_impute.*.tsv')
print(in_files[:5])


# In[12]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('tissue_impute.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics = tissue_metrics.rename({'impute_samples': 'pretraining_sample_count'}, axis='columns')
tissue_metrics


# In[13]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='factor(supervised)')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Effects of Imputation on Multiclass Tissue Prediction')
plot += facet_grid('pretraining_sample_count ~ .')
plot


# In[14]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='factor(pretraining_sample_count)')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Effects of Imputation on Multiclass Tissue Prediction')
plot


# ## Adding BioBERT Embeddings

# In[9]:


in_files = glob.glob('../../results/all-tissue-biobert*.tsv')
print(in_files[:5])


# In[10]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('biobert.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics = tissue_metrics.rename({'impute_samples': 'pretraining_sample_count'}, axis='columns')
tissue_metrics['supervised'] = tissue_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')

tissue_metrics


# In[11]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='supervised')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Multiclass Tissue Prediction With Expression + BioBERT')
plot


# In[ ]:





# ## Sample Split Positive Control

# In[2]:


in_files = glob.glob('../../results/sample-split.*.tsv')
print(in_files[:5])


# In[3]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('biobert.')[-1]
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['supervised'] = tissue_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')

tissue_metrics


# In[4]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='is_pretrained')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Transfer Learning Sample Positive Control')
plot += facet_grid('supervised ~ .')
plot


# In[17]:


single_run_df = tissue_metrics[tissue_metrics['seed'] == '1']
single_run_df.head()


# In[18]:


plot = ggplot(single_run_df, aes(x='train_count', y='balanced_accuracy', color='is_pretrained')) 
#plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Sample Control Single Run Points')
plot += facet_grid('supervised ~ .')
plot


# In[ ]:





# ## Study Split Positive Control

# In[2]:


in_files = glob.glob('../../results/study-split.*.tsv')
print(in_files[:5])


# In[3]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('biobert.')[-1]
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['supervised'] = tissue_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')

tissue_metrics


# In[4]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='is_pretrained')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Transfer Learning Study Positive Control')
plot += facet_grid('supervised ~ .')
plot


# In[5]:


single_run_df = tissue_metrics[tissue_metrics['seed'] == '1']
single_run_df.head()


# In[6]:


plot = ggplot(single_run_df, aes(x='train_count', y='balanced_accuracy', color='is_pretrained')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Study Level Single Run Points')
plot += facet_grid('supervised ~ .')
plot


# ## Tissue Split

# In[7]:


in_files = glob.glob('../../results/tissue-split.*.tsv')
print(in_files[:5])


# In[8]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('biobert.')[-1]
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['supervised'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['supervised'] = tissue_metrics['supervised'].str.replace('pytorch_supervised', 'three_layer_net')

tissue_metrics


# In[9]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='is_pretrained')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Cross-Tissue Pretraining')
plot += facet_grid('supervised ~ .')
plot


# In[5]:


single_run_df = tissue_metrics[tissue_metrics['seed'] == '1']
single_run_df.head()


# In[6]:


plot = ggplot(single_run_df, aes(x='train_count', y='balanced_accuracy', color='is_pretrained')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Cross-tissue Single Run Points')
plot += facet_grid('supervised ~ .')
plot

