#!/usr/bin/env python
# coding: utf-8

# ## Bad LR Performance
# This notebook pokes at the results to try to figure out what is causing LR methods to do worse than random when the linear signal is removed

# In[66]:


import collections
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
import sklearn.metrics
from plotnine import *

from saged.utils import split_sample_names, create_dataset_stat_df, get_dataset_stats, parse_map_file


# In[3]:


top_five_tissues = ['Blood', 'Breast', 'Stem_Cell', 'Cervix', 'Brain']
#top_five_tissues = ['Blood', 'Breast', 'Stem_Cell']

combo_iterator = itertools.combinations(top_five_tissues, 2)
tissue_pairs = [pair for pair in combo_iterator]
tissue_pairs[:3]


# In[4]:


in_files = []
for pair in tissue_pairs:
    in_files.extend(glob.glob('../../results/{}.{}.*signal_removed*.tsv'.format(pair[0], pair[1])))
len(in_files)


# In[5]:


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
        correction_method = 'signal_removed'
    elif 'study_corrected' in path:
        correction_method = 'study_corrected'
    
    new_df['supervised'] = model
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
run_results['supervised'] = run_results['supervised'].str.replace('pytorch_supervised', 'three_layer_net')
run_results


# In[8]:


run_results['val_encoders'].value_counts()


# In[27]:


sample_model_predictions = {}
model_predictions = {}
# Process dataframe rows
for i, row in run_results.iterrows():
    model = row['supervised']
    
    if model not in model_predictions:
        model_predictions[model] = {'predicted': [], 'true': []}
    
    encoder_string = row['val_encoders']
    try:
        encoder = json.loads(encoder_string)
    except TypeError:
        continue
    decoder = {number: label for label, number in encoder.items()}
    
    samples = row['val samples'].strip().split(',')
    
    predictions = row['val_predictions'].strip().split(',')
    truth = row['val_true_labels'].strip().split(',')
    pred_labels = []
    
    for prediction in predictions:
        if int(prediction) in decoder:
            pred_labels.append(decoder[int(prediction)])
        else:
            # https://github.com/greenelab/saged/issues/58
            pred_labels.append('invalid_index')
            
    predictions = pred_labels
    
    truth = [decoder[int(t)] for t in truth]
    
    assert len(truth) == len(predictions)
    
    model_predictions[model]['predicted'].extend(predictions)
    model_predictions[model]['true'].extend(truth)
    
    for sample, prediction, true_label in zip(samples, predictions, truth):
        if sample not in sample_model_predictions:
            sample_model_predictions[sample] = {}
        if model not in sample_model_predictions[sample]:
            sample_model_predictions[sample][model] = {'correct': 0, 'total': 0, 
                                                       'predictions': [], 'true_label': true_label}
        
        assert sample_model_predictions[sample][model]['true_label'] == true_label
        
        sample_model_predictions[sample][model]['predictions'].append(prediction)
        sample_model_predictions[sample][model]['total'] += 1
        
        correct = (prediction == true_label)
        if correct:
            sample_model_predictions[sample][model]['correct'] += 1


# In[39]:


def row_norm(row):
    new_row = row / row.sum()
    return new_row

def create_confusion_df(model_predictions, model, norm=True):
    predicted = model_predictions[model]['predicted']
    true = model_predictions[model]['true']

    confusion_matrix = sklearn.metrics.confusion_matrix(true, predicted)
    
    tissue_set = set(true).union(set(predicted))
    tissues = sorted(list(tissue_set))
    
    if norm:
        confusion_matrix = np.apply_along_axis(row_norm, axis=1, arr=confusion_matrix)
    confusion_df = pd.DataFrame(confusion_matrix, index = [l for l in tissues],
                                columns = [l for l in tissues])
    
    confusion_df['true_tissue'] = confusion_df.index
    confusion_df = confusion_df.dropna()
    return confusion_df


# In[47]:


confusion_df = create_confusion_df(model_predictions, 'deep_net', norm=True)


# In[50]:


melted_df = confusion_df.melt(id_vars='true_tissue', var_name='pred_tissue')
melted_df['log_value'] = np.log(melted_df['value'])
melted_df['percent'] = (melted_df['value'] * 100).round(1)


# In[54]:


plot = ggplot(melted_df, aes(x='true_tissue', y='pred_tissue', fill='percent',)) 
plot += geom_tile() 
plot += geom_text(aes(label='percent'))
plot += theme(axis_text_x=element_text(rotation=270, hjust=1))
plot += ggtitle('five layer net confusion matrix')
plot


# In[78]:


confusion_df = create_confusion_df(model_predictions, 'three_layer_net', norm=True)


# In[79]:


melted_df = confusion_df.melt(id_vars='true_tissue', var_name='pred_tissue')
melted_df['log_value'] = np.log(melted_df['value'])
melted_df['percent'] = (melted_df['value'] * 100).round(1)


# In[80]:


plot = ggplot(melted_df, aes(x='true_tissue', y='pred_tissue', fill='percent',)) 
plot += geom_tile() 
plot += geom_text(aes(label='percent'))
plot += theme(axis_text_x=element_text(rotation=270, hjust=1))
plot += ggtitle('Three layer net confusion matrix')
plot


# In[81]:


confusion_df = create_confusion_df(model_predictions, 'pytorch_lr', norm=True)


# In[82]:


melted_df = confusion_df.melt(id_vars='true_tissue', var_name='pred_tissue')
melted_df['log_value'] = np.log(melted_df['value'])
melted_df['percent'] = (melted_df['value'] * 100).round(1)


# In[83]:


plot = ggplot(melted_df, aes(x='true_tissue', y='pred_tissue', fill='percent',)) 
plot += geom_tile() 
plot += geom_text(aes(label='percent'))
plot += theme(axis_text_x=element_text(rotation=270, hjust=1))
plot += ggtitle('Pytorch_lr confusion matrix')
plot


# In[62]:


confusion_df = create_confusion_df(model_predictions, 'logistic_regression', norm=True)


# In[63]:


melted_df = confusion_df.melt(id_vars='true_tissue', var_name='pred_tissue')
melted_df['log_value'] = np.log(melted_df['value'])
melted_df['percent'] = (melted_df['value'] * 100).round(1)


# In[64]:


plot = ggplot(melted_df, aes(x='true_tissue', y='pred_tissue', fill='percent',)) 
plot += geom_tile() 
plot += geom_text(aes(label='percent'))
plot += theme(axis_text_x=element_text(rotation=270, hjust=1))
plot += ggtitle('Sklearn LR confusion matrix')
plot


# ## Prediction distributions

# In[92]:


deep_net_predicted = collections.Counter(model_predictions['deep_net']['predicted'])
deep_net_true = predicted = collections.Counter(model_predictions['deep_net']['true'])
print(deep_net_predicted, deep_net_true)


# In[93]:


three_layer_net_predicted = collections.Counter(model_predictions['three_layer_net']['predicted'])
three_layer_net_true = predicted = collections.Counter(model_predictions['three_layer_net']['true'])
print(three_layer_net_predicted, three_layer_net_true)


# In[94]:


pytorch_lr_predicted = collections.Counter(model_predictions['pytorch_lr']['predicted'])
pytorch_lr_true = predicted = collections.Counter(model_predictions['pytorch_lr']['true'])
print(pytorch_lr_predicted, pytorch_lr_true)


# In[95]:


skl_lr_predicted = collections.Counter(model_predictions['logistic_regression']['predicted'])
skl_lr_true = predicted = collections.Counter(model_predictions['logistic_regression']['true'])
print(skl_lr_predicted, skl_lr_true)


# In[101]:


pred_df = pd.DataFrame([deep_net_predicted, three_layer_net_predicted, pytorch_lr_predicted, skl_lr_predicted])
pred_df['model'] = ['five_layer_net', 'three_layer_net', 'pytorch_lr', 'skl_lr']
pred_df['pred_or_true'] = 'pred'
pred_df


# In[102]:


true_df = pd.DataFrame([deep_net_true, three_layer_net_true, pytorch_lr_true, skl_lr_true])
true_df['model'] = ['five_layer_net', 'three_layer_net', 'pytorch_lr', 'skl_lr']
true_df['pred_or_true'] = 'true'
true_df


# In[106]:


all_df = pd.concat([pred_dict, true_dict])
all_df


# In[120]:


plot = ggplot(all_df, aes(x='factor(pred_or_true)', y='Blood', fill='model'))
plot += geom_bar(stat='identity')
plot += facet_grid('model ~ .')
print(plot)


# In[121]:


plot = ggplot(all_df, aes(x='factor(pred_or_true)', y='Breast', fill='model'))
plot += geom_bar(stat='identity')
plot += facet_grid('model ~ .')
print(plot)


# In[122]:


plot = ggplot(all_df, aes(x='factor(pred_or_true)', y='Stem Cell', fill='model'))
plot += geom_bar(stat='identity')
plot += facet_grid('model ~ .')
print(plot)


# In[123]:


plot = ggplot(all_df, aes(x='factor(pred_or_true)', y='Cervix', fill='model'))
plot += geom_bar(stat='identity')
plot += facet_grid('model ~ .')
print(plot)


# In[124]:


plot = ggplot(all_df, aes(x='factor(pred_or_true)', y='Brain', fill='model'))
plot += geom_bar(stat='identity')
plot += facet_grid('model ~ .')
print(plot)


# ## Conclusion
# There doesn't seem to be anything significantly different between the prediction distributions of the deep models and the linear models. The linear ones are just wrong more regardless of class.
# 
# Actually the differences between the five layer net and three layer net in terms of their prediction strategy are interesting (they perform better with respect to different classes).
# 
# My working hypothesis is now that the linear models are learning spurious correlations in the training set that cause them to overfit the test set
