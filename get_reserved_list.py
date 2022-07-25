#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import pip
pckgs = ['pandas','sqlalchemy','datetime','Office365-REST-Python-Client']
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', '--proxy=http://proxy.iil.intel.com:911', package])

for p in pckgs:
    import_or_install(p)


# In[3]:


from py_topping.data_connection.sharepoint import lazy_SP365
import pandas as pd
from sqlalchemy import create_engine
from datetime import date
today = date.today()


# In[4]:


### used this guide https://faun.pub/quick-etl-with-python-part-1-download-files-from-sharepoint-online-40bf23711662


# In[5]:


# Import library
# Create connection
sp = lazy_SP365(site_url = 'https://intel.sharepoint.com/sites/IPAMDataQuality'
                   , client_id = 'd41c4361-de53-4b02-a260-516a66728c62'
                   , client_secret = 'ZFxlzvWnWvAC7to9irYF0k7RKnjmSNEVC8W/da37RuQ=')
# Create download path from download URL
download_path = sp.create_link('https://intel.sharepoint.com/:x:/r/sites/IPAMDataQuality/Shared%20Documents/General/2022_IPAM_range_mgmt_data_quaility/get-it-clean%20Reserved%20Networks/IPAM_CLEAN_RESERVED_NETWORKS.xlsx?d=w50030ee0da31460eb0db753ad7d91a32&csf=1&web=1&e=lS0n7o')
# Download file


# In[6]:


sp.download(sharepoint_location = download_path, local_location = rf"IPAM_CLEAN_RESERVED_NETWORKS.xlsx")


# In[11]:


reserved_df = pd.read_excel("IPAM_CLEAN_RESERVED_NETWORKS.xlsx").rename({"added by (WWID)":"added_by"}, axis=1)
reserved_df


# In[9]:


download_path = sp.create_link('https://intel.sharepoint.com/:x:/r/sites/IPAMDataQuality/Shared%20Documents/General/2022_IPAM_range_mgmt_data_quaility/get-it-clean%20Reserved%20Networks/IPAM_CLEAN_DELETED_NETWORKS.xlsx?d=w50030ee0da31460eb0db753ad7d91a32&csf=1&web=1&e=lS0n7o')
sp.download(sharepoint_location = download_path, local_location = rf"IPAM_CLEAN_DELETED_NETWORKS.xlsx")


# In[10]:


deleted_df = pd.read_excel("IPAM_CLEAN_DELETED_NETWORKS.xlsx").rename({"added by (WWID)":"added_by"}, axis=1)
deleted_df


# In[12]:


engine = create_engine('postgresql://cppmapi_so:dKqZpJmKz2n11Dq@postgres5561-lb-fm-in.iglb.intel.com:5433/cppmapi')


# ### upload reserved

# In[13]:


old_data = pd.read_sql_query('select * from public."IPAM_reserved_ranges"', engine)


# In[14]:


merge_data = reserved_df.merge(old_data, on='Range', how='left',suffixes=( '_new','_old'))


# In[15]:


merge_data.date = merge_data.date.fillna(today)
merge_data.date = pd.to_datetime(merge_data.date).dt.date


# In[16]:


merge_data = merge_data.sort_values('date', ascending=False).drop_duplicates(subset='Range', keep='first').reset_index(drop=True)


# In[17]:


for idx,row in merge_data.iterrows():
    if (row.added_by_old == row.added_by_new) and (row.reason_old == row.reason_new):
        continue
    else:
        merge_data.at[idx,'added_by_old'] = row.added_by_new
        merge_data.at[idx,'reason_old'] = row.reason_new
        merge_data.at[idx,'date'] = today


# In[18]:


merge_data = merge_data[['Range','added_by_old','date','reason_old']].rename({'added_by_old':'added_by','reason_old':'reason'}, axis=1) 


# In[19]:


merge_data.to_sql(name='IPAM_reserved_ranges', con=engine, if_exists = 'replace', index=False, method='multi',  chunksize = 2097 // len(merge_data.columns))


# In[20]:


merge_data


# ### upload deleted

# In[21]:


merge_data = deleted_df.merge(old_data, on='Range', how='left',suffixes=( '_new','_old'))


# In[22]:


merge_data.date = merge_data.date.fillna(today)
merge_data.date = pd.to_datetime(merge_data.date).dt.date


# In[23]:


merge_data = merge_data.sort_values('date', ascending=False).drop_duplicates(subset='Range', keep='first').reset_index(drop=True)


# In[24]:


for idx,row in merge_data.iterrows():
    if (row.added_by_old == row.added_by_new) and (row.reason_old == row.reason_new):
        continue
    else:
        merge_data.at[idx,'added_by_old'] = row.added_by_new
        merge_data.at[idx,'reason_old'] = row.reason_new
        merge_data.at[idx,'date'] = today


# In[25]:


merge_data = merge_data[['Range','added_by_old','date','reason_old']].rename({'added_by_old':'added_by','reason_old':'reason'}, axis=1) 


# In[26]:


merge_data.to_sql(name='IPAM_deleted_ranges', con=engine, if_exists = 'replace', index=False, method='multi',  chunksize = 2097 // len(merge_data.columns))


# In[ ]:




