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


# In[2]:


data_dir = '../../data'


# In[3]:


metadata_path = os.path.join(data_dir, 'aggregated_metadata.json')
metadata = None
with open(metadata_path) as json_file:
    metadata = json.load(json_file)
sample_metadata = metadata['samples']


# In[4]:


experiments = metadata['experiments']
sample_to_study = {}
for study in experiments:
    for accession in experiments[study]['sample_accession_codes']:
        sample_to_study[accession] = study


# In[5]:


in_files = glob.glob('../../results/impute.*')
print(in_files[:5])


# In[6]:


metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    
    model_info = path.strip('.tsv').split('results/impute.')[-1]
    model_info = model_info.split('.')
        
    if len(model_info) == 4:
        unsupervised_model = model_info[0]
        supervised_model = model_info[1]
        seed = model_info[3]
    else:
        unsupervised_model = 'untransformed'
        supervised_model = model_info[0]
        seed = model_info[2]
             
    new_df['unsupervised'] = unsupervised_model
    new_df['supervised'] = supervised_model
        
    metrics = pd.concat([metrics, new_df])
    
metrics = metrics.reset_index()
metrics['trial'] = metrics.index // 10
metrics.head()


# ## A note on the "trial" variable:
# The loss between different training/val dataset splits is difficult to compare. As a result, I've grouped models together by whether they were produced by the same dataset split/seed. All ten points within the same trial correspond to models trained on increasingly large subsets of the same original training dataset, validated on identical validation sets.

# In[7]:


ggplot(metrics, aes(x='train sample count', y='val_loss', color='factor(trial)')) + geom_line()


# ## Center trials
# To make the trend easier to see, center trial to have a mean of zero

# In[8]:


mean_centered_metrics = metrics.copy()
mean_centered_metrics['val_loss'] = (mean_centered_metrics['val_loss'] - 
                                     mean_centered_metrics.groupby('trial').transform('mean')['val_loss'])
mean_centered_metrics.head()


# In[9]:


plot = ggplot(mean_centered_metrics, aes(x='train sample count', y='val_loss', color='factor(trial)'))
plot += geom_line()
plot += ggtitle('The relationship between sample count and val set loss')
print(plot)


# In[10]:


plot = ggplot(mean_centered_metrics, aes(x='train sample count', y='val_loss', color='factor(trial)')) 
plot += geom_point()
plot += ggtitle('The relationship between sample count and val set loss')
print(plot)


# In[11]:


plot = ggplot(mean_centered_metrics, aes(x='train sample count', y='val_loss')) 
plot += geom_point()
plot += geom_smooth(method='loess')
plot += ggtitle('The relationship between sample count and val set loss')
print(plot)


# ## Evaluate Transfering Models

# In[12]:


in_files = glob.glob('../../results/transfer.*')
print(in_files[:5])


# In[17]:


metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    
    model_info = path.strip('.tsv').split('results/impute.')[-1]
    model_info = model_info.split('.')        

    supervised_model = model_info[4]
    seed = model_info[6]
    disease = model_info[3]
             
    new_df['supervised'] = supervised_model
    new_df['seed'] = seed
    new_df['disease'] = disease
        
    metrics = pd.concat([metrics, new_df])
    
metrics = metrics.reset_index()
metrics['trial'] = metrics.index // 20
metrics


# In[18]:


sepsis_df = metrics[metrics['disease'] == 'sepsis']
tb_df = metrics[metrics['disease'] == 'tb']


# In[19]:


ggplot(sepsis_df, aes(x='train sample count', y='balanced_accuracy', color='factor(trial)')) + geom_line()


# In[20]:


plot = ggplot(sepsis_df, aes(x='train sample count', y='balanced_accuracy', color='factor(impute_samples)')) 
plot += geom_point()
plot += geom_smooth(method='loess')
plot += ggtitle('The relationship between sample count and val set loss')
print(plot)


# In[21]:


ggplot(tb_df, aes(x='train sample count', y='balanced_accuracy', color='factor(trial)')) + geom_line()


# In[22]:


plot = ggplot(tb_df, aes(x='train sample count', y='balanced_accuracy', color='factor(impute_samples)')) 
plot += geom_point()
plot += geom_smooth(method='loess')
plot += ggtitle('The relationship between sample count and val set loss')
print(plot)

