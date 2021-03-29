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
    
    #print(new_df.head())
    #break
    model_info = path.strip('.tsv').split('results/impute.')[-1]
    model_info = model_info.split('.')
    print(model_info)
        
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
metrics


# In[7]:


ggplot(metrics, aes(x='train sample count', y='val_loss', color='factor(trial)')) + geom_line()


# ## Center trials
# To make the trend easier to see, center each set of ten training iterations 

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

