
# coding: utf-8

# In[1]:

import graphlab as gl
import numpy as np
import random as rnd
import math
from __future__ import division
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
gl.canvas.set_target('ipynb')


# In[231]:

cities = gl.SFrame('India_500_Cities.csv')
simple_features = ['effective_literacy_rate_total', 'sex_ratio', 'total_graduates']
label = 'name_of_city'


# In[232]:

cities1 = gl.SFrame(cities['name_of_city', 'effective_literacy_rate_total', 'sex_ratio', 'total_graduates'])
cities1['total_graduates'] = cities1['total_graduates']*1.
cities1['sex_ratio'] = cities1['sex_ratio']*1.


# In[233]:

cities1


# In[204]:




# # Simple model

# In[276]:

simple_model = gl.nearest_neighbors.create(cities1, features = simple_features, label = 'name_of_city',verbose=False)
simple_model_cosine = gl.nearest_neighbors.create(cities1, features = simple_features, label = 'name_of_city', distance = 'cosine',verbose=False)


# In[227]:

chennai = cities1[cities1['name_of_city'] == 'Chennai']

hyd = cities1[cities1['name_of_city'] == 'Greater Hyderabad']

bang = cities1[cities1['name_of_city'] == 'Bengaluru']


# ## Chennai query

# In[206]:

print simple_model.query(chennai, verbose=False)
print simple_model_cosine.query(chennai,verbose=False)


# ## Bangalore query

# In[207]:

print simple_model.query(bang, verbose=False)
print simple_model_cosine.query(bang,verbose=False)


# ## Hyderabad query

# In[208]:

print simple_model.query(hyd, verbose=False)
print simple_model_cosine.query(hyd,verbose=False)


# # Create some useful functions

# # Feature Scaling, Get query point row data

# In[270]:

def feature_scaling(data, features, scaling_type):
    scaled_data = {}
    scaled_data['name_of_city'] = data['name_of_city']
    
    if scaling_type == 1:
        # L2 Euclidean norm scaling - Inner dot product scalar
        
        for feature in features:
          
            norm = np.dot(data[feature], data[feature]) 
            scaled_data[feature] = data[feature]*1./norm
                    
    elif scaling_type == 2:
        # Mean/Std deviation based scaling
        for feature in features:
            
            scaled_data[feature] = ((data[feature]-data[feature].mean())*1.)/data[feature].std()
    
    
    return gl.SFrame(scaled_data)

def get_query(query_name, data):
    return data[data['name_of_city'] == query_name]


# In[272]:

# print 'Raw data', cities1[0:2]
# norm_features = feature_scaling(cities1, simple_features, 1)
# print 'Normalized data', norm_features[0:2]
# std_features = feature_scaling(cities1, simple_features, 2)
# print 'Standardized data', std_features[0:2]
# chn_norm = get_query('Chennai', norm_features)
# hyd_norm = get_query('Greater Hyderabad', norm_features)
# bang_norm = get_query('Bengaluru', norm_features)


# In[250]:




# # Learn models

# In[274]:

simple_model = gl.nearest_neighbors.create(cities1, features = simple_features, label = 'name_of_city', distance = 'euclidean', verbose=False)
simple_model_cosine = gl.nearest_neighbors.create(cities1, features = simple_features, label = 'name_of_city', distance = 'cosine',verbose=False)

norm_model = gl.nearest_neighbors.create(norm_features, features = simple_features, label = 'name_of_city', distance = 'euclidean', verbose=False)
norm_model_cosine = gl.nearest_neighbors.create(norm_features, features = simple_features, label = 'name_of_city', distance = 'cosine', verbose=False)

std_model = gl.nearest_neighbors.create(std_features, features = simple_features, label = 'name_of_city', distance = 'euclidean', verbose=False)
std_model_cosine = gl.nearest_neighbors.create(std_features, features = simple_features, label = 'name_of_city', distance = 'cosine', verbose=False)


# ## Chennai query

# In[275]:

print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print 'Raw features Euclidean', simple_model.query(get_query('Chennai', cities1), verbose=False)
print 'Raw features Cosine', simple_model_cosine.query(get_query('Chennai', cities1), verbose=False)

print 'Normalized Euclidean features', norm_model.query(get_query('Chennai', feature_scaling(cities1, simple_features, 1)),verbose=False)
print 'Normalized Cosine features', norm_model_cosine.query(get_query('Chennai', feature_scaling(cities1, simple_features, 1)), verbose=False)

print 'Standardized Euclidean features', std_model.query(get_query('Chennai', feature_scaling(cities1, simple_features, 2)), verbose=False)
print 'Standardized Cosine features', std_model_cosine.query(get_query('Chennai', feature_scaling(cities1, simple_features, 2)), verbose=False)


# In[ ]:



