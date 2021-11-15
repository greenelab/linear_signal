#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import glob
import json
import os

import numpy as np
import pandas as pd
import pickle
from plotnine import *
import sklearn.metrics
from sklearn.decomposition import PCA
import umap


# In[2]:


data_dir = '../../data'
metadata_path = os.path.join(data_dir, 'recount_metadata.tsv')
metadata_df = pd.read_csv(metadata_path, sep='\t')
metadata_df = metadata_df.drop_duplicates()


# In[3]:


metadata_df


# In[4]:


sample_to_study = dict(zip(metadata_df['external_id'], metadata_df['study']))


# In[5]:


label_file = os.path.join(data_dir, 'recount_sample_to_label.pkl')
with open(label_file, 'rb') as in_file:
    sample_to_label = pickle.load(in_file)


# In[6]:


metadata_df['label'] = metadata_df['external_id'].map(sample_to_label)


# ## Load the results from a sample-split experiment

# In[7]:


in_files = glob.glob('../../results/sample-split.*.tsv')
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


sample_model_predictions = {}
model_predictions = {}
# Process dataframe rows
for i, row in tissue_metrics[tissue_metrics['is_pretrained'] == 'not_pretrained'].iterrows():
    model = row['supervised']
    
    if model not in model_predictions:
        model_predictions[model] = {'predicted': [], 'true': []}
    
    encoder_string = row['val_encoders']
    encoder = json.loads(encoder_string)
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


# In[10]:


def row_norm(row):
    new_row = row / row.sum()
    return new_row

def create_confusion_df(model_predictions, norm=True):
    predicted = model_predictions['deep_net']['predicted']
    true = model_predictions['deep_net']['true']

    confusion_matrix = sklearn.metrics.confusion_matrix(true, predicted, labels=list(encoder.keys()))

    
    if norm:
        confusion_matrix = np.apply_along_axis(row_norm, axis=1, arr=confusion_matrix)
    confusion_df = pd.DataFrame(confusion_matrix, index = [l for l in encoder.keys()],
                                columns = [l for l in encoder.keys()])
    
    confusion_df['true_tissue'] = confusion_df.index
    confusion_df = confusion_df.dropna()
    return confusion_df


# In[11]:


confusion_df = create_confusion_df(model_predictions, norm=False)


# In[12]:


melted_df = confusion_df.melt(id_vars='true_tissue', var_name='pred_tissue')
melted_df['log_value'] = np.log(melted_df['value'])


# In[13]:


plot = ggplot(melted_df, aes(x='true_tissue', y='pred_tissue', fill='log_value',)) 
plot += geom_tile() 
plot += geom_text(aes(label='value'), size=6)
plot += theme(axis_text_x=element_text(rotation=270, hjust=1))
plot += ggtitle('Confusion matrix of all samples')
plot


# In[14]:


confusion_df = create_confusion_df(model_predictions, norm=True)


# In[15]:


melted_df = confusion_df.melt(id_vars='true_tissue', var_name='pred_tissue')
melted_df['log_value'] = np.log(melted_df['value'])
melted_df['percent'] = (melted_df['value'] * 100).round(1)


# In[16]:


plot = ggplot(melted_df, aes(x='true_tissue', y='pred_tissue', fill='percent',)) 
plot += geom_tile() 
plot += geom_text(aes(label='percent'), size=6)
plot += theme(axis_text_x=element_text(rotation=270, hjust=1))
plot += ggtitle('Confusion matrix of all samples')
plot


# In[17]:


data_dict_list = []

for sample in sample_model_predictions:
    data_dict = {}
    data_dict['sample'] = sample
    for model in sample_model_predictions[sample]:
        total = sample_model_predictions[sample][model]['total']
        correct = sample_model_predictions[sample][model]['correct']
        acc = correct / total
        percent_wrong = 100 * (1 - acc)
        
        data_dict['{}_total'.format(model)] =  total 
        data_dict['{}_correct'.format(model)] = correct
        data_dict['{}_acc'.format(model)] = acc
        data_dict['{}_percent_wrong'.format(model)] = percent_wrong
    data_dict_list.append(data_dict)
        
acc_df = pd.DataFrame(data_dict_list)


# In[18]:


acc_df


# In[19]:


plot = ggplot(acc_df, aes(x='deep_net_percent_wrong', y='pytorch_lr_percent_wrong'))
plot += geom_bin2d()
plot


# In[20]:


plot = ggplot(acc_df, aes(x='deep_net_percent_wrong', y='logistic_regression_percent_wrong'))
plot += geom_bin2d()
plot


# ## What do the individual distributions look like?

# In[21]:


acc_df['pytorch_lr_acc'].plot.hist()


# In[22]:


acc_df['three_layer_net_acc'].plot.hist()


# In[23]:


acc_df['deep_net_acc'].plot.hist()


# In[24]:


acc_df['logistic_regression_acc'].plot.hist()


# ### Results
# The pytorch models have more variation than the sklearn LR model. In general, there do seem to be some samples that are harder to classify. Is this a quirk of the dataset, or something that happens across datasets?

# ## Look at samples misclassified by all models in every case

# In[25]:


hard_samples = []
for sample in sample_model_predictions:
    good_prediction = False
    for model in sample_model_predictions[sample]:
        correct = sample_model_predictions[sample][model]['correct']
        if correct > 0:
            good_prediction = True
    
    if not good_prediction:
        hard_samples.append(sample)


# In[26]:


print(len(hard_samples))

studies = set([sample_to_study[sample] for sample in hard_samples])
print(len(studies))

print(studies)


# In[27]:


print(hard_samples[:5])


# ### Are all samples in a study with hard samples unclassifiable?

# In[28]:


samples_in_bad_studies = 0
for sample, study in sample_to_study.items():
    if study in studies:
        samples_in_bad_studies += 1

print(len(hard_samples), samples_in_bad_studies, len(hard_samples) / samples_in_bad_studies)


# For each sample that is universally incorrectly classified, there are six other in the same study that are fine

# ### Are there any hints in the metadata about why the samples are hard to classify?

# In[29]:


metadata_df = metadata_df.set_index('external_id')


# In[30]:


metadata_df = metadata_df[metadata_df.index.notnull()]


# In[31]:


hard_sample_df = metadata_df.loc[hard_samples]
hard_sample_df


# In[32]:


metadata_df['recount_seq_qc.avgq'] = pd.to_numeric(metadata_df['recount_seq_qc.avgq'], errors='coerce')
metadata_df = metadata_df[metadata_df['recount_seq_qc.avgq'].notnull()]
metadata_df


# In[33]:


metadata_df['recount_seq_qc.avgq'] = metadata_df['recount_seq_qc.avgq'].astype(float)
metadata_df['recount_seq_qc.avgq'].plot.hist(title='AvgQ Distributon')


# In[34]:


hard_sample_df['recount_seq_qc.avgq'] = hard_sample_df['recount_seq_qc.avgq'].astype(float)
hard_sample_df['recount_seq_qc.avgq'].plot.hist(title='AvgQ Distributon in Hard Samples')


# ## Do the hard samples stand out in a UMAP embedding?

# In[35]:


expression_df = pd.read_pickle("../../data/no_scrna_tpm.pkl")


# In[36]:


# Running UMAP on 200k samples just makes a giant oval
is_hard = [sample in hard_samples for sample in expression_df.index]
expression_df['is_hard'] = is_hard

easy_df = expression_df[expression_df['is_hard'] == False]
easy_df = easy_df.sample(n=4000, random_state=42)
hard_df = expression_df[expression_df['is_hard']]
sampled_df = pd.concat([easy_df, hard_df])

# Remove column to extract expression for UMAP
is_hard = sampled_df['is_hard']
sampled_df = sampled_df.drop(['is_hard'], axis='columns')

sampled_df


# In[37]:


expression_matrix = sampled_df.to_numpy()
expression_matrix.shape


# In[38]:


reducer = umap.UMAP(transform_seed=42, random_state=42, n_components=2)


# In[39]:


get_ipython().run_cell_magic('time', '', 'expression_umap = reducer.fit_transform(expression_matrix)')


# In[40]:


umap_df = pd.DataFrame(expression_umap, index=sampled_df.index, columns=['UMAP1', 'UMAP2'])


# In[41]:


umap_df


# In[42]:


umap_df['is_hard'] = is_hard


# In[43]:


umap_df


# In[44]:


plot = ggplot(umap_df, aes(x='UMAP1', y='UMAP2', fill='is_hard'))
plot += geom_point()
plot += ggtitle('UMAP Embedding of all samples')
print(plot)


# ## There seems to be a reasonable high-d linear dividing line since LR works well. Maybe PCA makes more sense for visualizing?

# In[45]:


reducer = PCA(random_state=42, n_components=2)


# In[46]:


get_ipython().run_cell_magic('time', '', 'expression_pca = reducer.fit_transform(expression_matrix)')


# In[47]:


pca_df = pd.DataFrame(expression_pca, index=sampled_df.index, columns=['UMAP1', 'UMAP2'])


# In[48]:


pca_df['is_hard'] = is_hard


# In[49]:


plot = ggplot(pca_df, aes(x='UMAP1', y='UMAP2', fill='is_hard'))
plot += geom_point()
plot += ggtitle('PCA Embedding of all samples')
print(plot)


# ## Do the hard samples stand out compared to other samples with the same label?

# In[50]:


hard_sample_df['label'].value_counts()


# In[51]:


metadata_df['label'].value_counts()


# The samples that are consistently mislabeled tend to be from the less common classes.
# 
# Potential causes:
# 1. There are fewer studies for those samples, so the inter-study differences are stronger (shouldn't be the case though since we're looking at an example with built-in leakage of study info between train and test sets)
# 2. There are fewer samples for the classes the hard samples are drawn from, so they are inherently harder predicton problems
# 3. Class imbalance causes these classes to consistently be predicted to be one of the more common classes (possibly in addition to the above)
# 4. Some samples may just be different from other samples in their class due to technical noise or vague labels

# In[52]:


expression_df = expression_df.drop('is_hard', axis='columns')
is_hard = [sample in hard_samples for sample in metadata_df.index]
metadata_df['is_hard'] = is_hard


# ## Are the hard samples study-specific?

# In[53]:


# Num samples per study
metadata_df[metadata_df['label'] == 'Umbilical Cord'].groupby('study').count()['is_hard']


# In[54]:


# Num hard samples per study
metadata_df[metadata_df['label'] == 'Umbilical Cord'].groupby('study').sum()['is_hard']


# In[55]:


metadata_df[(metadata_df['study'] == 'SRP033491') & (metadata_df['label'] == 'Umbilical Cord')]


# ### Manual inspection
# It looks like there are lots of studies with hard samples and umbilical cord labels. A few stand out:
# 
# SRP021214 - 6 samples, 6 of which are consistently misclassified; all of which are HeLa cells not actual umbilical cord
# 
# SRP033491 - 8 samples, 4 of which are hard. It's hard to tell what tissue they're from based on the metadata, so the curation could be off. Also they all have really low errq (~15)
# 

# ## Do the hard samples show up on a UMAP?

# In[56]:


expression_samples = set(expression_df.index)

uc_df = metadata_df[metadata_df['label'] == 'Umbilical Cord']
uc_samples = [sample for sample in uc_df.index if sample in expression_samples]

uc_expression = expression_df.loc[uc_samples]

uc_expression


# In[57]:


expression_matrix = uc_expression.to_numpy()


# In[58]:


reducer = umap.UMAP(transform_seed=42, random_state=42, n_components=2)
uc_umap = reducer.fit_transform(expression_matrix)


# In[59]:


umap_df = pd.DataFrame(uc_umap, columns=['UMAP1', 'UMAP2'], index=uc_expression.index)
umap_df


# In[60]:


is_hard = [sample in hard_samples for sample in umap_df.index]
umap_df['is_hard'] = is_hard
umap_df['study'] = metadata_df.loc[umap_df.index]['study']
umap_df['platform'] = metadata_df.loc[umap_df.index]['sra.platform_model']
umap_df['date'] = metadata_df.loc[umap_df.index]['sra.run_published']


umap_df


# In[61]:


plot = ggplot(umap_df, aes(x='UMAP1', y='UMAP2', color='is_hard'))
plot += geom_point()
plot += ggtitle('UMAP embedding of umbilical cord samples')
print(plot)


# In[62]:


plot = ggplot(umap_df, aes(x='UMAP1', y='UMAP2', color='study'))
plot += geom_point()
plot += ggtitle('UMAP embedding of umbilical cord samples')
print(plot)


# In[63]:


plot = ggplot(umap_df, aes(x='UMAP1', y='UMAP2', color='platform'))
plot += geom_point()
plot += ggtitle('UMAP embedding of umbilical cord samples')
print(plot)


# In[64]:


plot = ggplot(umap_df, aes(x='UMAP1', y='UMAP2', color='date'))
plot += geom_point()
plot += ggtitle('UMAP embedding of umbilical cord samples')
print(plot)


# ### PCA?

# In[65]:


reducer = PCA(random_state=42, n_components=2)
uc_pca = reducer.fit_transform(expression_matrix)


# In[66]:


pca_df = pd.DataFrame(uc_pca, columns=['PC1', 'PC2'], index=uc_expression.index)
pca_df


# In[67]:


is_hard = [sample in hard_samples for sample in umap_df.index]
pca_df['is_hard'] = is_hard
pca_df['study'] = metadata_df.loc[umap_df.index]['study']
pca_df['platform'] = metadata_df.loc[umap_df.index]['sra.platform_model']
pca_df['date'] = metadata_df.loc[umap_df.index]['sra.run_published']


umap_df


# In[68]:


plot = ggplot(pca_df, aes(x='PC1', y='PC2', color='is_hard'))
plot += geom_point()
plot += ggtitle('PCA embedding of umbilical cord samples')
print(plot)


# In[69]:


plot = ggplot(pca_df, aes(x='PC1', y='PC2', color='study'))
plot += geom_point()
plot += ggtitle('PCA embedding of umbilical cord samples')
print(plot)


# In[70]:


plot = ggplot(pca_df, aes(x='PC1', y='PC2', color='platform'))
plot += geom_point()
plot += ggtitle('PCA embedding of umbilical cord samples')
print(plot)


# In[71]:


plot = ggplot(pca_df, aes(x='PC1', y='PC2', color='date'))
plot += geom_point()
plot += ggtitle('PCA embedding of umbilical cord samples')
print(plot)


# ## Sequencer?
# There doesn't seem an expression difference that causes all the bad samples to cluster together, but there does seem to be a disproportionate amount of hard to classify samples from the HiSeq 2000. Does this hold in general/in other tissue types?

# In[72]:


hard_df = metadata_df[metadata_df['is_hard']]


# In[73]:


hard_df['sra.platform_model'].value_counts()


# In[74]:


hard_df['sra.platform_model'].value_counts() / len(hard_df)


# In[75]:


metadata_df['sra.platform_model'].value_counts() / len(metadata_df)


# ### Sequencer Results
# 
# There seems to be a marked increase in hard-to-classify Hi-seq 2000 samples and decrease in Hi-seq 2500 samples. Is there a reason for this?
# 

# In[76]:


hard_studies = set(hard_df['study'])
hard_study_df = metadata_df[metadata_df['study'].isin(hard_studies)]


# In[77]:


hard_study_df['sra.platform_model'].value_counts() 


# In[78]:


hard_study_df['sra.platform_model'].value_counts() / len(hard_study_df)


# ### More sequencer results
# The enrichment for Hi-seq 2000 may just be because the studies with hard samples are themselves enriched for Hi-seq 2000. 
# The sequencer probably may not be meaningful, but this is still interesting. It feels like the samples are drawn at random from the studies, yielding the given distribution.
# 
# Unclear whether this is true though, the studies themselves seem to to be heavily skewed.

# ## Remaining questions:
# - Is it related to sparsity?
# - Is there something correlated with sequencer that we'd expect to cause these results?
# - Stuff below

# ## Error analysis on hard samples

# In[79]:


sample_model_predictions = {}
model_predictions = {}
# Process dataframe rows
for i, row in tissue_metrics[tissue_metrics['is_pretrained'] == 'not_pretrained'].iterrows():
    model = row['supervised']
    
    if model not in model_predictions:
        model_predictions[model] = {'predicted': [], 'true': [], 'sample': []}
    
    encoder_string = row['val_encoders']
    encoder = json.loads(encoder_string)
    decoder = {number: label for label, number in encoder.items()}
    
    samples = row['val samples'].strip().split(',')
    
    predictions = row['val_predictions'].strip().split(',')
    truth = row['val_true_labels'].strip().split(',')
    pred_labels = []
    true_labels = []

    for prediction, true_label, sample in zip(predictions, truth, samples):
        
        if sample not in hard_samples:
            continue
        
        if int(prediction) in decoder:
            pred_labels.append(decoder[int(prediction)])
        else:
            # https://github.com/greenelab/saged/issues/58
            pred_labels.append('invalid_index')
            
        true_labels.append(decoder[int(true_label)])
               
    predictions = pred_labels
    truth = true_labels
        
    assert len(truth) == len(predictions)
    
    model_predictions[model]['predicted'].extend(predictions)
    model_predictions[model]['true'].extend(truth)
    model_predictions[model]['sample'].extend(samples)


# In[80]:


confusion_df = create_confusion_df(model_predictions)


# In[81]:


melted_df = confusion_df.melt(id_vars='true_tissue', var_name='pred_tissue')
melted_df['log_value'] = np.log(melted_df['value'])
melted_df['percent'] = (melted_df['value'] * 100).round(1)


# In[82]:


plot = ggplot(melted_df, aes(x='true_tissue', y='pred_tissue', fill='percent',)) 
plot += geom_tile() 
plot += geom_text(aes(label='percent'), size=6)
plot += theme(axis_text_x=element_text(rotation=270, hjust=1))
plot += ggtitle('Confusion matrix of samples that are always misclassified')
plot


# ## Conclusion (for now)
# Fixing class imbalance helped some, but the weird distribution of errors is still there. It may be due to low read quality, but it's unclear

# ### A few remaining questions:
# - What is the correlation in errors between different runs of the same model?
# - Are these correlations affected by the amount of data used?
