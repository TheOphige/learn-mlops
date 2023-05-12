#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json


# In[2]:


url= 'http://localhost:9696/predict'


# In[13]:


with open('XTestSet.json', 'rb') as f_in:
    persons= json.load(f_in)

persons


# In[4]:


persons['0']


# In[24]:


response = requests.post(url, json= persons['0']).json()
response


# In[26]:


if response['diabetic'] == True:
    print('You are Diabetic, take medicine %s' % ('xyz-123'))
else:
        print('You are not Diabetic')


# In[27]:


for person in persons:
    response = requests.post(url, json= persons[person]).json()
    print(response)
    if response['diabetic'] == True:
        print('You are Diabetic, take medicine %s' % ('xyz-123'))
    else:
        print('You are not Diabetic')


# In[ ]:




