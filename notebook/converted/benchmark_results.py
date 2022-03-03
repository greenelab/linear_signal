#!/usr/bin/env python
# coding: utf-8

# # Benchmark Results
# This notebook visualizes the output from the different models on different classification problems

# In[1]:


import collections
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
import plotnine
from plotnine import *

from saged.utils import split_sample_names, create_dataset_stat_df, get_dataset_stats, parse_map_file


# ## Binary Prediction

# In[2]:


top_five_tissues = ['Blood', 'Breast', 'Stem_Cell', 'Cervix', 'Brain']

combo_iterator = itertools.combinations(top_five_tissues, 2)
tissue_pairs = [pair for pair in combo_iterator]
tissue_pairs[:3]


# In[3]:


in_files = []
for pair in tissue_pairs:
    in_files.extend(glob.glob('../../results/{}.{}.*.tsv'.format(pair[0], pair[1])))
len(in_files)


# In[4]:


run_results = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    run_info = os.path.basename(path).split('.')
    
    tissue1 = run_info[0]
    tissue2 = run_info[1]
    
    model_and_seed = run_info[2].split('-')[0]
    seed = model_and_seed.split('_')[-1]
    model = '_'.join(model_and_seed.split('_')[:-1])
    
    correction_method = 'unmodified'
    if 'signal_removed' in path:
        if 'sample_level' in path:
            correction_method = 'signal_removed_sample_level'
        else:
            correction_method = 'signal_removed'
    elif 'study_corrected' in path:
        correction_method = 'study_corrected'
    elif 'split_signal' in path:
        correction_method = 'split_signal'
    
    new_df['Model'] = model
    new_df['seed'] = seed
    if tissue1 < tissue2:
        new_df['tissue1'] = tissue1
        new_df['tissue2'] = tissue2
    else:
        new_df['tissue1'] = tissue2
        new_df['tissue2'] = tissue1
    new_df['correction_method'] = correction_method
    new_df['pair'] = new_df['tissue1'] + '-' + new_df['tissue2'] 
    
    run_results = pd.concat([run_results, new_df])
    
run_results['train_count'] = run_results['train sample count']
run_results['Model'] = run_results['Model'].str.replace('pytorch_supervised', 'three_layer_net')
run_results['Model'] = run_results['Model'].str.replace('deep_net', 'five_layer_net')


# ## Pairwise comparisons

# In[5]:


plot_df = run_results[run_results['correction_method'] == 'unmodified']
plot_df = plot_df[plot_df['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()

plot += geom_hline(yintercept=.5, linetype='dashed')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot += facet_grid(['tissue1', 'tissue2'], scales='fixed')
plot += ggtitle('Recount3 Binary Classification')
plot += theme(axis_text_x=element_text(rotation=90, hjust=.25))
plot.save('../../figures/recount_binary.svg')
plot


# In[6]:


plot_df = run_results[run_results['correction_method'] == 'signal_removed']
plot_df = plot_df[plot_df['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += geom_hline(yintercept=.5, linetype='dashed')


plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot += facet_grid(['tissue1', 'tissue2'], scales='fixed')
plot += ggtitle('Recount3 Binary Classification After Signal Removal')
plot += theme(axis_text_x=element_text(rotation=90, hjust=.25))
plot.save('../../figures/recount_binary_signal_removed.svg')
plot


# In[7]:


plot_df = run_results[run_results['correction_method'] == 'split_signal']
plot_df = plot_df[plot_df['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]

plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += geom_hline(yintercept=.5, linetype='dashed')

plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot += facet_grid(['tissue1', 'tissue2'], scales='fixed')
plot += ggtitle('Recount3 Classification After Fold-Separated Signal Removal')
plot += theme(axis_text_x=element_text(rotation=90, hjust=.25))
plot.save('../../figures/recount_binary_split_limma.svg')
plot


# ## All Tissue Predictions

# In[8]:


in_files = glob.glob('../../results/all-tissue.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[9]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('all-tissue.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[10]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 

plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=1/21, linetype='dashed')
plot += ggtitle('Recount3 21-Class Tissue Prediction')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/recount_multiclass.svg')
plot


# ### All tissue signal removed

# In[11]:


in_files = glob.glob('../../results/all-tissue.*.tsv')
in_files = [f for f in in_files if 'signal_removed' in f]
print(in_files[:5])


# In[12]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('all-tissue.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-3])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-3]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[13]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 

plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=1/21, linetype='dashed')
plot += ggtitle('Recount3 21-Class Tissue Prediction Signal Removed')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/recount_multiclass_signal_removed.svg')
plot


# ## All tissue sample split

# In[14]:


in_files = glob.glob('../../results/all-tissue_sample-split*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[15]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('all-tissue_sample-split.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
    new_df['split'] = 'sample'
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])


# In[16]:


more_files = glob.glob('../../results/all-tissue.*')
more_files = [f for f in more_files if 'be_corrected' not in f]
print(more_files[:5])


# In[17]:


for path in more_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('all-tissue.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
    new_df['split'] = 'study'
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[18]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth(aes(linetype='split'))
plot += ggtitle('Recount3 Effects of Data Splitting')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/recount_multiclass_sample_split.svg')
plot


# ## Pretraining

# In[19]:


in_files = glob.glob('../../results/study-split.*.tsv')
print(in_files[:5])


# In[20]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv')
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')
#tissue_metrics['Model'] = tissue_metrics['Model'] + '_' + tissue_metrics['is_pretrained']
tissue_metrics['pretrained'] = True
tissue_metrics.loc[tissue_metrics['is_pretrained'] == 'not_pretrained', 'pretrained'] = False
tissue_metrics


# In[21]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth(aes(linetype='pretrained'))
plot += ggtitle('Recount3 Multiclass Classification with Pretraining')
plot += scale_linetype_manual(['dashed', 'solid'])
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/recount_pretraining.svg')
plot


# ## Sex Prediction

# In[22]:


in_files = glob.glob('../../results/sample-split-sex-prediction.*.tsv')
print(in_files[:5])


# In[23]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv')
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')

tissue_metrics


# In[24]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += ggtitle('Metadata sex prediction (samplewise split)')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/sex_prediction_samplewise.svg')
plot


# ### Study Level

# In[25]:


in_files = glob.glob('../../results/study-split-sex-prediction.*.tsv')
print(in_files[:5])


# In[26]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('biobert.')[-1]
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')

tissue_metrics


# In[27]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += ggtitle('Metadata sex prediction')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/sex_prediction_studywise.svg')
plot


# ### Study level signal removed

# In[28]:


in_files = glob.glob('../../results/sex-prediction-signal-removed.*.tsv')
print(in_files[:5])


# In[29]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sex-prediction-signal-removed.')[-1]
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[30]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += ggtitle('Metadata sex prediction signal removed')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/sex_prediction_signal_removed.svg')
plot


# ### Study level split signal removed

# In[31]:


in_files = glob.glob('../../results/sex-prediction-split-signal.*.tsv')
print(in_files[:5])


# In[32]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sex-prediction-split-signal.')[-1]
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')

tissue_metrics


# In[33]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += ggtitle('Metadata sex prediction split signal removal')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/sex_prediction_split_signal.svg')
plot


# ## GTEx All Tissues

# In[34]:


in_files = glob.glob('../../results/gtex-all-tissue.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[35]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('all-tissue.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[36]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
#plot += geom_point(alpha=.2)
plot += ggtitle('All Tissue Prediction GTEx Data')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/gtex_multiclass.svg')
plot


# In[37]:


in_files = glob.glob('../../results/gtex-all-tissue-signal-removed.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[38]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('gtex-all-tissue-signal-removed.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[39]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += ggtitle('GTEx Multiclass Signal Removed')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/gtex_multiclass_signal_removed.svg')
plot


# ## Binary Predictions

# In[40]:


top_five_tissues = ['Blood', 'Brain', 'Skin', 'Esophagus', 'Blood_Vessel']

combo_iterator = itertools.combinations(top_five_tissues, 2)
tissue_pairs = [pair for pair in combo_iterator]
tissue_pairs[:3]


# In[41]:


in_files = []
for pair in tissue_pairs:
    in_files.extend(glob.glob('../../results/gtex.{}.{}.*.tsv'.format(pair[0], pair[1])))
len(in_files)


# In[42]:


run_results = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    run_info = os.path.basename(path).split('.')
        
    tissue1 = run_info[1]
    tissue2 = run_info[2]
    
    model_and_seed = run_info[3].split('-')[0]
    seed = model_and_seed.split('_')[-1]
    model = '_'.join(model_and_seed.split('_')[:-1])
    
    correction_method = 'unmodified'
    if 'signal_removed' in path:
        if 'sample_level' in path:
            correction_method = 'signal_removed_sample_level'
        else:
            correction_method = 'signal_removed'
    elif 'study_corrected' in path:
        correction_method = 'study_corrected'
    elif 'split_signal' in path:
        correction_method = 'split_signal'
    
    new_df['Model'] = model
    new_df['seed'] = seed
    if tissue1 < tissue2:
        new_df['tissue1'] = tissue1
        new_df['tissue2'] = tissue2
    else:
        new_df['tissue1'] = tissue2
        new_df['tissue2'] = tissue1
    new_df['correction_method'] = correction_method
    new_df['pair'] = new_df['tissue1'] + '-' + new_df['tissue2'] 
    
    run_results = pd.concat([run_results, new_df])
    
run_results['train_count'] = run_results['train sample count']
run_results['Model'] = run_results['Model'].str.replace('pytorch_supervised', 'three_layer_net')
run_results['Model'] = run_results['Model'].str.replace('deep_net', 'five_layer_net')


# In[43]:


plot_df = run_results[run_results['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot_df = plot_df[plot_df['correction_method'] == 'unmodified']
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += facet_grid(['tissue1', 'tissue2'], scales='free')
plot += ggtitle('Pairwise comparisons no be correction')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/gtex_pairwise.svg')
plot


# In[44]:


in_files = []
for pair in tissue_pairs:
    in_files.extend(glob.glob('../../results/gtex-signal-removed.{}.{}.*.tsv'.format(pair[0], pair[1])))
len(in_files)


# In[45]:


run_results = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    run_info = os.path.basename(path).split('.')
        
    tissue1 = run_info[1]
    tissue2 = run_info[2]
    
    model_and_seed = run_info[3].split('-')[0]
    seed = model_and_seed.split('_')[-1]
    model = '_'.join(model_and_seed.split('_')[:-1])
        
    new_df['Model'] = model
    new_df['seed'] = seed
    if tissue1 < tissue2:
        new_df['tissue1'] = tissue1
        new_df['tissue2'] = tissue2
    else:
        new_df['tissue1'] = tissue2
        new_df['tissue2'] = tissue1
    new_df['pair'] = new_df['tissue1'] + '-' + new_df['tissue2'] 
    
    run_results = pd.concat([run_results, new_df])
    
run_results['train_count'] = run_results['train sample count']
run_results['Model'] = run_results['Model'].str.replace('pytorch_supervised', 'three_layer_net')
run_results['Model'] = run_results['Model'].str.replace('deep_net', 'five_layer_net')


# In[46]:


plot_df = run_results[run_results['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += facet_grid(['tissue1', 'tissue2'], scales='free')
plot += ggtitle('Gtex pairwise comparisons with signal removed')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/gtex_pairwise_signal_removed.svg')
plot


# ## Simulated data 

# In[47]:


in_files = glob.glob('../../results/sim-data.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[48]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sim-data.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[49]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Simulated data')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/sim_data.svg')
plot


# In[50]:


in_files = glob.glob('../../results/sim-data-signal-removed.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[51]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sim-data-signal-removed.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[52]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += geom_smooth()
plot += ggtitle('Simulated data after signal removal')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/sim_data_signal_removed.svg')
plot


# In[53]:


in_files = glob.glob('../../results/no-signal-sim-data.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[54]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sim-data.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[55]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += geom_smooth()
plot += ggtitle('Simulated data without signal')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/no_signal_sim.svg')
plot


# In[56]:


in_files = glob.glob('../../results/no-signal-sim-data-signal-removed.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[57]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sim-data-signal-removed.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1]) 
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[58]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += geom_smooth()
plot += ggtitle('No signal simulated data + signal removal')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/no_signal_sim_signal_removed.svg')
plot


# In[59]:


in_files = glob.glob('../../results/no-signal-sim-data-split-signal.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[60]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sim-data-split-signal.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[61]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += geom_smooth()
plot += ggtitle('No Signal Simulation with Split Signal Removal')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/no_signal_sim_split_signal.svg')
plot


# ### Split signal removal (be correction in train and val sets separately)

# In[62]:


in_files = glob.glob('../../results/sim-data-split-signal.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[63]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sim-data-split-signal.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')
tissue_metrics


# In[64]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += geom_smooth()
plot += ggtitle('Simulated data prediction - split signal removal')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/sim_split_signal_removal.svg')
plot


# ## Linear signal only simulation

# In[65]:


in_files = glob.glob('../../results/linear-sim-data.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[66]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('linear-sim-data.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[67]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += geom_smooth()
plot += ggtitle('Simulated data prediction - linear features only')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/linear_sim.svg')
plot


# In[68]:


in_files = glob.glob('../../results/linear-sim-data-signal-removed.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[69]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('sim-data-signal-removed.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')
tissue_metrics


# In[70]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += geom_smooth()
plot += ggtitle('Simulated data prediction - linear vars, signal removed')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/linear_sim_signal_removed.svg')
plot


# In[71]:


in_files = glob.glob('../../results/linear-sim-data-split-signal.*.tsv')
in_files = [f for f in in_files if 'be_corrected' not in f]
print(in_files[:5])


# In[72]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('linear-sim-data-split-signal.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[73]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += geom_smooth()
plot += ggtitle('Simulated data prediction - linear vars, split signal removal')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/linear_sim_split_signal_removal.svg')
plot


# ## Unused Plots 

# In[74]:


top_five_tissues = ['Blood', 'Breast', 'Stem_Cell', 'Cervix', 'Brain']

combo_iterator = itertools.combinations(top_five_tissues, 2)
tissue_pairs = [pair for pair in combo_iterator]
tissue_pairs[:3]


# In[75]:


in_files = []
for pair in tissue_pairs:
    in_files.extend(glob.glob('../../results/{}.{}.*.tsv'.format(pair[0], pair[1])))
len(in_files)


# In[76]:


run_results = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    run_info = os.path.basename(path).split('.')
    
    tissue1 = run_info[0]
    tissue2 = run_info[1]
    
    model_and_seed = run_info[2].split('-')[0]
    seed = model_and_seed.split('_')[-1]
    model = '_'.join(model_and_seed.split('_')[:-1])
    
    correction_method = 'unmodified'
    if 'signal_removed' in path:
        if 'sample_level' in path:
            correction_method = 'signal_removed_sample_level'
        else:
            correction_method = 'signal_removed'
    elif 'study_corrected' in path:
        correction_method = 'study_corrected'
    elif 'split_signal' in path:
        correction_method = 'split_signal'
    
    new_df['Model'] = model
    new_df['seed'] = seed
    if tissue1 < tissue2:
        new_df['tissue1'] = tissue1
        new_df['tissue2'] = tissue2
    else:
        new_df['tissue1'] = tissue2
        new_df['tissue2'] = tissue1
    new_df['correction_method'] = correction_method
    new_df['pair'] = new_df['tissue1'] + '-' + new_df['tissue2'] 
    
    run_results = pd.concat([run_results, new_df])
    
run_results['train_count'] = run_results['train sample count']
run_results['Model'] = run_results['Model'].str.replace('pytorch_supervised', 'three_layer_net')
run_results['Model'] = run_results['Model'].str.replace('deep_net', 'five_layer_net')
run_results


# In[77]:


plot_df = run_results[run_results['correction_method'] == 'study_corrected']
plot_df = plot_df[plot_df['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += geom_hline(yintercept=.5, linetype='dashed')

plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot += facet_grid(['tissue1', 'tissue2'], scales='fixed')
plot += ggtitle('Recount3 Binary Classification After Study Correction')
plot += theme(axis_text_x=element_text(rotation=90, hjust=.25))
plot


# In[78]:


plot_df = run_results[run_results['correction_method'] == 'signal_removed_sample_level']
plot_df = plot_df[plot_df['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Pairwise comparisons tissue signal removed (sample level split)')

plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot += facet_grid(['tissue1', 'tissue2'], scales='fixed')
plot += ggtitle('Samplewise-split After Signal Removal')
plot += theme(axis_text_x=element_text(rotation=90, hjust=.25))
plot
plot


# ## Tissue Prediction

# In[79]:


in_files = glob.glob('../../results/Blood.Breast.*.tsv')
print(in_files[:5])


# In[80]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    run_info = os.path.basename(path).split('.')
    
    tissue1 = run_info[0]
    tissue2 = run_info[1]
    
    model_and_seed = run_info[2].split('-')[0]
    seed = model_and_seed.split('_')[-1]
    model = '_'.join(model_and_seed.split('_')[:-1])
    
    correction_method = 'unmodified'
    if 'signal_removed' in path:
        correction_method = 'signal_removed'
    elif 'study_corrected' in path:
        correction_method = 'study_corrected'
    
    new_df['Model'] = model
    new_df['seed'] = seed
    new_df['tissue1'] = tissue1
    new_df['tissue2'] = tissue2
    new_df['correction_method'] = correction_method
    
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics


# In[81]:


plot_df = tissue_metrics[tissue_metrics['correction_method'] == 'unmodified']
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Blood vs Breast Tissue Prediction')
plot


# ### BE Corrected binary tissue classification

# In[82]:


in_files = glob.glob('../../results/Blood.Breast.*.tsv')
in_files = [f for f in in_files if 'study_corrected' in f]
print(in_files[:5])


# In[83]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('Breast.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics


# In[84]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='Model')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += ggtitle('Blood vs Breast Tissue Prediction')
plot


# ## Imputation pretraining

# In[85]:


in_files = glob.glob('../../results/tissue_impute.*.tsv')
print(in_files[:5])


# In[86]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('tissue_impute.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics = tissue_metrics.rename({'impute_samples': 'pretraining_sample_count'}, axis='columns')
tissue_metrics


# In[87]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='factor(Model)')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Effects of Imputation on Multiclass Tissue Prediction')
plot += facet_grid('pretraining_sample_count ~ .')
plot


# In[88]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='factor(pretraining_sample_count)')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Effects of Imputation on Multiclass Tissue Prediction')
plot


# ## Sample split positive control

# In[89]:


in_files = glob.glob('../../results/sample-split.*.tsv')
print(in_files[:5])


# In[90]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('biobert.')[-1]
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')

tissue_metrics


# In[91]:


plot = ggplot(tissue_metrics, aes(x='train_count', y='balanced_accuracy', color='is_pretrained')) 
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Transfer Learning Sample Positive Control')
plot += facet_grid('Model ~ .')
plot


# In[92]:


single_run_df = tissue_metrics[tissue_metrics['seed'] == '1']
single_run_df.head()


# In[93]:


plot = ggplot(single_run_df, aes(x='train_count', y='balanced_accuracy', color='is_pretrained')) 
#plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Sample Control Single Run Points')
plot += facet_grid('Model ~ .')
plot


# ## Tissue Signal Removed

# ### Sample level

# In[94]:


in_files = glob.glob('../../results/sample-split-signal-removed.*.tsv')
print(in_files[:5])


# In[95]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv')
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')


# In[96]:


viz_df = tissue_metrics[tissue_metrics['is_pretrained'] == 'not_pretrained']
plot = ggplot(viz_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Sample level tissue corrected')
plot


# ### Study level

# In[97]:


in_files = glob.glob('../../results/study-split-signal-removed.*.tsv')
print(in_files[:5])


# In[98]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('biobert.')[-1]
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')


# In[99]:


viz_df = tissue_metrics[tissue_metrics['is_pretrained'] == 'not_pretrained']
plot = ggplot(viz_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Study level tissue corrected')
plot


# ## Study Signal Removed

# ### Sample level

# In[100]:


in_files = glob.glob('../../results/sample-split-study-corrected.*.tsv')
print(in_files[:5])


# In[101]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('biobert.')[-1]
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')


# In[102]:


viz_df = tissue_metrics[tissue_metrics['is_pretrained'] == 'not_pretrained']
plot = ggplot(viz_df, aes(x='train_count', y='balanced_accuracy', color='supervised'))
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Sample level study corrected')
plot


# ### Study level

# In[103]:


in_files = glob.glob('../../results/study-split-study-corrected.*.tsv')
print(in_files[:5])


# In[104]:


tissue_metrics = pd.DataFrame()

for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('biobert.')[-1]
    model_info = model_info.split('.')
    model_info = model_info[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:2])
             
    new_df['Model'] = supervised_model
    
    new_df['seed'] = model_info[-1]
            
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')


# In[105]:


viz_df = tissue_metrics[tissue_metrics['is_pretrained'] == 'not_pretrained']
plot = ggplot(viz_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += ggtitle('Study-based split, study corrected')
plot


# ## Mutation prediction

# In[106]:


in_files = glob.glob('../../results/tcga-binary.*.tsv')
print(in_files[:5])


# In[107]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('tcga-binary.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model.split('.')[-1]
    new_df['gene'] = supervised_model.split('.')[0]
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[108]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += geom_smooth()
#plot += geom_point(alpha=.2)
plot += facet_wrap(['gene'])
plot += ggtitle('TCGA mutation prediction')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/tcga.svg')
plot


# In[109]:


in_files = glob.glob('../../results/tcga-binary-signal-removed.*.tsv')
print(in_files[:5])


# In[110]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('tcga-binary-signal-removed.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model.split('.')[-1]
    new_df['gene'] = supervised_model.split('.')[0]
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[111]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += facet_wrap(['gene'])
plot += ggtitle('TCGA signal removed')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/tcga_signal_removed.svg')
plot


# In[112]:


in_files = glob.glob('../../results/tcga-binary-split-signal.*.tsv')
print(in_files[:5])


# In[113]:


tissue_metrics = pd.DataFrame()
for path in in_files:
    new_df = pd.read_csv(path, sep='\t')
    model_info = path.strip('.tsv').split('tcga-binary-split-signal.')[-1]
    model_info = model_info.split('_')
        
    supervised_model = '_'.join(model_info[:-1])
             
    new_df['Model'] = supervised_model.split('.')[-1]
    new_df['gene'] = supervised_model.split('.')[0]
    
    new_df['seed'] = model_info[-1]
        
    tissue_metrics = pd.concat([tissue_metrics, new_df])
    
tissue_metrics['train_count'] = tissue_metrics['train sample count']

tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('pytorch_supervised', 'three_layer_net')
tissue_metrics['Model'] = tissue_metrics['Model'].str.replace('deep_net', 'five_layer_net')


# In[114]:


plot_df = tissue_metrics[tissue_metrics['Model'].isin(['three_layer_net', 'five_layer_net', 'pytorch_lr'])]
plot = ggplot(plot_df, aes(x='train_count', y='balanced_accuracy', color='Model'))
plot += geom_hline(yintercept=.5, linetype='dashed')
plot += geom_smooth()
plot += geom_point(alpha=.2)
plot += facet_wrap(['gene'])
plot += ggtitle('TCGA split signal')
plot += xlab('Train sample count')
plot += ylab('Balanced Accuracy (Val Set)')
plot.save('../../figures/tcga_split_signal.svg')
plot

