#!/usr/bin/env python
# coding: utf-8

# # Evaluation of Batch Effect Correction
# This notebook evaluates the extent to which limma and ComBat correct for batch effects by plotting the results and seeing to what extent the inter-batch differences remain in the first two PCs

# In[1]:


import collections
import json
import os
import pickle

import pandas as pd
import sklearn
from plotnine import ggplot, geom_point, aes
from tqdm.notebook import tqdm

from saged import utils


# In[2]:


def return_unlabeled():
    # For use in a defaultdict
    return 'unlabeled'


# In[3]:


data_dir = '../../data/'
map_file = os.path.join(data_dir, 'sample_classifications.pkl')

sample_to_label = utils.parse_map_file(map_file)
sample_to_label = collections.defaultdict(return_unlabeled, sample_to_label)

with open(map_file, 'rb') as in_file:
    label_to_sample = pickle.load(in_file)[0]


# In[4]:


metadata_path = os.path.join(data_dir, 'aggregated_metadata.json')
metadata = None
with open(metadata_path) as json_file:
    metadata = json.load(json_file)


# ## Get blood samples from dataset
# 

# In[5]:


sample_ids = utils.get_blood_sample_ids(metadata, sample_to_label)
print(len(sample_ids))


# In[6]:


compendium_path = os.path.join(data_dir, 'HOMO_SAPIENS.tsv')                                                             

# Not all labeled samples show up in the compendium, which causes pandas to panic. To fix this  
# we have to take the intersection of the accessions in sample_ids and the accessions in the       
# compendium                                                                                    
header_ids = None                                                                               
with open(compendium_path) as in_file:                                                          
    header = in_file.readline()                                                                 
    header_ids = header.split('\t')                                                             

valid_sample_ids = [id_ for id_ in sample_ids if id_ in header_ids]  
valid_sample_ids = set(valid_sample_ids)
print(len(valid_sample_ids))


# ## Read in expression data
# Now that we know which samples we're interested in, let's read those samples from the compendium.
# This is doable with pandas, but subsetting a dataframe is extremely slow for some reason.
# Manually writing the subsetted tsv to a file and loading it should be substantially faster and
# less memory intensive

# In[7]:


out_path = os.path.join(data_dir, 'subset_compendium.tsv')


# In[8]:


with open(compendium_path) as compendium_file:
    with open(out_path, 'w') as out_file:
        header = compendium_file.readline().strip().split('\t')
        # Start with zero included to keep all gene names
        col_indices_to_keep = [i+1 for i in range(len(header)) if header[i] in valid_sample_ids]
        col_indices_to_keep = [0] + col_indices_to_keep
        
        # Print header, keeping empty index column as in the original compendium
        header = [''] + header
        # You can't index a list with a list of indices in python, so we have to 
        # Use the __getitem__ function for each index
        header_string = '\t'.join(map(header.__getitem__, col_indices_to_keep))
        out_file.write('{}\n'.format(header_string))
        
        for line in tqdm(compendium_file):
            line = line.strip().split('\t')
            out_string = '\t'.join(map(line.__getitem__, col_indices_to_keep))
            out_file.write('{}\n'.format(out_string))
            


# In[9]:


compendium_df = pd.read_csv(out_path, sep='\t', index_col=0)


# ### Get Sample Metadata

# In[10]:


experiments = metadata['experiments']
sample_to_study = {}
for study in experiments:
    for accession in experiments[study]['sample_accession_codes']:
        sample_to_study[accession] = study


# ### Create PCA Embedding of all data

# In[11]:


reducer = sklearn.decomposition.PCA(n_components=2)
compendium_df = compendium_df.transpose()
pca_embedding = reducer.fit_transform(compendium_df.values)
print(reducer.explained_variance_ratio_)

pca_embedding.shape


# In[12]:


compendium_df.head()


# In[13]:


embedding_df = pd.DataFrame.from_dict({'sample': list(compendium_df.index),
                                       'pc1': list(pca_embedding[:,0]),
                                       'pc2': list(pca_embedding[:,1]),
                                      })


# In[14]:


sample_metadata = metadata['samples']

platforms = []
for sample in embedding_df['sample']:
    platforms.append(sample_metadata[sample]['refinebio_platform'].lower())
    
embedding_df['platform'] = platforms


# In[15]:


embedding_df['study'] = embedding_df['sample'].map(sample_to_study)
embedding_df['label'] = embedding_df['sample'].map(sample_to_label)


# In[16]:


embedding_df.head()


# In[17]:


ggplot(embedding_df, aes(x='pc1', y='pc2', color='study')) + geom_point()


# In[18]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='platform')) + geom_point()


# In[19]:


ggplot(embedding_df, aes(x='pc1', y='pc2', color='label')) + geom_point()


# In[20]:


study_vector = embedding_df['study'].values
platform_vector = embedding_df['platform'].values


# In[21]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[22]:


expression_values = compendium_df.transpose().values


# In[23]:


expression_values.shape


# ## Correct batches with limma

# In[24]:


study_corrected = utils.run_limma(expression_values, study_vector)
platform_corrected = utils.run_limma(expression_values, platform_vector)
full_corrected = utils.run_limma(expression_values, study_vector, platform_vector)


# In[25]:


# Run batch effect correction
# Create PCA embedding of corrected data
# Add to the dataframe
# Plot and compare uncorrected and batch corrected 
study_corrected = study_corrected.transpose()
platform_corrected = platform_corrected.transpose()
full_corrected = full_corrected.transpose()


# In[26]:


study_corrected_pcs = reducer.fit_transform(study_corrected)
print(reducer.explained_variance_ratio_)
platform_corrected_pcs = reducer.fit_transform(platform_corrected)
print(reducer.explained_variance_ratio_)
full_corrected_pcs = reducer.fit_transform(full_corrected)
print(reducer.explained_variance_ratio_)


# In[27]:


embedding_df['study_pc1'] = study_corrected_pcs[:,0]
embedding_df['study_pc2'] = study_corrected_pcs[:,1]
embedding_df['platform_pc1'] = platform_corrected_pcs[:,0]
embedding_df['platform_pc2'] = platform_corrected_pcs[:,1]
embedding_df['full_correction_pc1'] = full_corrected_pcs[:,0]
embedding_df['full_correction_pc2'] = full_corrected_pcs[:,1]


# In[28]:


embedding_df.head()


# ## Plot by platform

# In[29]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='platform')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='study_pc1', y='study_pc2', color='platform')) + geom_point()
print(plot)


# In[30]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='platform')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='platform_pc1', y='platform_pc2', color='platform')) + geom_point()
print(plot)


# In[31]:


ggplot(embedding_df, aes(x='full_correction_pc1', y='full_correction_pc2', color='platform')) + geom_point()


# ## Plot by study

# In[32]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='study')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='study_pc1', y='study_pc2', color='study')) + geom_point()
print(plot)


# In[33]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='study')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='platform_pc1', y='platform_pc2', color='study')) + geom_point()
print(plot)


# ## Plot by label

# In[34]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='label')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='study_pc1', y='study_pc2', color='label')) + geom_point()
print(plot)


# In[35]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='label')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='platform_pc1', y='platform_pc2', color='label')) + geom_point()
print(plot)


# ## Limma Results
# It looks like Limma does a good job of correcting for batch effects, but may overcorrect, especiall when accounting for all studies.
# 
# It's also worth noting that due to the tendancy of a study to use only one platform, correcting for study and platform is redundant

# ## Correct batches with ComBat

# In[36]:


study_corrected = utils.run_combat(expression_values, study_vector)
platform_corrected = utils.run_combat(expression_values, platform_vector)


# In[37]:


study_corrected = study_corrected.transpose()
platform_corrected = platform_corrected.transpose()


# In[38]:


study_corrected_pcs = reducer.fit_transform(study_corrected)
print(reducer.explained_variance_ratio_)
platform_corrected_pcs = reducer.fit_transform(platform_corrected)
print(reducer.explained_variance_ratio_)


# In[39]:


embedding_df['study_pc1'] = study_corrected_pcs[:,0]
embedding_df['study_pc2'] = study_corrected_pcs[:,1]
embedding_df['platform_pc1'] = platform_corrected_pcs[:,0]
embedding_df['platform_pc2'] = platform_corrected_pcs[:,1]


# In[40]:


embedding_df.head()


# ## Plot by platform

# In[41]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='platform')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='study_pc1', y='study_pc2', color='platform')) + geom_point()
print(plot)


# In[42]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='platform')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='platform_pc1', y='platform_pc2', color='platform')) + geom_point()
print(plot)


# ## Plot by study

# In[43]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='study')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='study_pc1', y='study_pc2', color='study')) + geom_point()
print(plot)


# In[44]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='study')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='platform_pc1', y='platform_pc2', color='study')) + geom_point()
print(plot)


# ## Plot by label

# In[45]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='label')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='study_pc1', y='study_pc2', color='label')) + geom_point()
print(plot)


# In[46]:


plot = ggplot(embedding_df, aes(x='pc1', y='pc2', color='label')) + geom_point()
print(plot)
plot = ggplot(embedding_df, aes(x='platform_pc1', y='platform_pc2', color='label')) + geom_point()
print(plot)


# ## ComBat Conclusions
# Combat appears to correct less severely than Limma does. As a result the labels look more separable after correction, but there also appears to be lingering interplatform differences after study and platform correction
