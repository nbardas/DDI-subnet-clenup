#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option("display.max_columns", None)
from sqlalchemy import create_engine
import xmltodict
import ipaddress as ia
from numpy import nan
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from datetime import date,timedelta
today = date.today()


# In[2]:


def ips_amount(row):
    net = ia.ip_network(row.Range)
    return net.num_addresses


# In[3]:


engine = create_engine('postgresql://cppmapi_so:dKqZpJmKz2n11Dq@postgres5561-lb-fm-in.iglb.intel.com:5433/cppmapi')
con = engine.connect()


# In[4]:


data_orig = pd.read_sql_query('select * from public."IPAM_full_fragmentation" where (is_leaf)', con=con)


# In[5]:


data_orig['ips_amount'] = data_orig.apply(ips_amount, axis=1)


# In[6]:


data_orig['sub_amount'] = 1


# In[7]:


data_orig[['Range', 'parent', 'sixteen_predecessor', 'is_discovered',
       'new_BuildingCode', 'new_environment', 'new_siteCode', 'new_campusCode',
       'is_top_environment', 'is_top_siteCode', 'is_top_campusCode','ips_amount','sub_amount']]


# In[8]:


envs = list(set(data_orig.new_environment))


# In[9]:


# env_dict = dict()
# for env in envs:
#     if env in ['none','multi']:
#         continue
#     temp = data_orig[data_orig.new_environment == env]
#     env_dict[env] = {"ranges": len(temp)}


# In[10]:


# x = data_orig[['ips_amount','sub_amount','is_discovered']].sort_values('ips_amount').groupby('is_discovered').sum()
# x.plot.pie(subplots=True,figsize=(30, 20),autopct="")


# In[11]:


# import matplotlib.pyplot as plt

# # make the pie circular by setting the aspect ratio to 1
# fig = plt.figure(figsize=plt.figaspect(1))
# fig.set_size_inches(7,7)
# values = list(x['ips_amount'])
# labels = list(x.index)

# def make_autopct(values):
#     def my_autopct(pct):
#         total = sum(values)
#         val = int(round(pct*total/100.0))
#         return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
#     return my_autopct

# plt.pie(values, labels=labels, autopct=make_autopct(values))
# plt.title("Number of IP addresses")
# plt.show()


# In[12]:



# # make the pie circular by setting the aspect ratio to 1
# fig = plt.figure(figsize=plt.figaspect(1))
# fig.set_size_inches(7,7)
# values = list(x['sub_amount'])
# labels = list(x.index)

# def make_autopct(values):
#     def my_autopct(pct):
#         total = sum(values)
#         val = int(round(pct*total/100.0))
#         return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
#     return my_autopct

# plt.pie(values, labels=labels, autopct=make_autopct(values))
# plt.title("Number of Subnets")
# plt.show()


# In[13]:


y = data_orig.drop(['is_leaf'], axis=1).sort_values('ips_amount').groupby(['new_environment','is_discovered']).sum()


# In[14]:


rows = []
for env in envs:
    if env in ['none','multi','IVL*','ICS','PENDING*','IVL']:
        continue
    tmp = y.loc[env]
    row = [env]
    try:
        row.append(tmp.loc['in use']['ips_amount'])
    except:
        row.append(0)
    try:
        row.append(tmp.loc['last seen old']['ips_amount'])
    except:
        row.append(0)    
    try:
        row.append(tmp.loc['never discovered']['ips_amount'])
    except:
        row.append(0)
    try:
        row.append(tmp.loc['free']['ips_amount'])
    except:
        row.append(0)
    try:
        row.append(tmp.loc['KEEP']['ips_amount'])
    except:
        row.append(0)
    rows.append(row)


# In[15]:


# create data
df = pd.DataFrame(rows, columns=['Environment', 'in use', 'last seen old', 'never discovered', 'free','KEEP']).sort_values('Environment')
# view data
env_ip_df = df[~df.Environment.str.contains(',')].sort_values("in use", ascending=False).reset_index(drop=True)
env_ip_df['collection_date'] = today
for col in ['in use', 'last seen old', 'never discovered','free','KEEP']:
    env_ip_df[col] = env_ip_df[col].astype(int)
display(env_ip_df)
  
# plot data in stack manner of bar type
env_ip_df.plot(x='Environment', kind='bar', stacked=True,title='IP addreses usage by environment', figsize=(20,3),width=0.7)


# In[16]:


rows = []
for env in envs:
    if env in ['none','multi','IVL*','ICS','PENDING*','IVL']:
        continue
    tmp = y.loc[env]
    row = [env]
    try:
        row.append(tmp.loc['in use']['sub_amount'])
    except:
        row.append(0)
    try:
        row.append(tmp.loc['last seen old']['sub_amount'])
    except:
        row.append(0)    
    try:
        row.append(tmp.loc['never discovered']['sub_amount'])
    except:
        row.append(0)
    try:
        row.append(tmp.loc['free']['sub_amount'])
    except:
        row.append(0)
    try:
        row.append(tmp.loc['KEEP']['sub_amount'])
    except:
        row.append(0)
    rows.append(row)


# In[17]:


# create data

df = pd.DataFrame(rows, columns=['Environment', 'in use', 'last seen old', 'never discovered', 'free','KEEP']).sort_values('Environment')
env_subnet_df = df[~df.Environment.str.contains(',')].sort_values("in use", ascending=False).reset_index(drop=True)
env_subnet_df['collection_date'] = today
for col in ['in use', 'last seen old', 'never discovered','free','KEEP']:
    env_subnet_df[col] = env_subnet_df[col].astype(int)
# view data
display(env_subnet_df)
  
# plot data in stack manner of bar type
env_subnet_df.plot(x='Environment', kind='bar', stacked=True,title='Subnet usage by environment', figsize=(20,3),width=0.7)


# In[18]:


last_updated = pd.read_sql_query('select max(collection_date) from public."IPAM_clenup_per_env_IP_count"', con=con)


# In[19]:


env_ip_df


# In[20]:


try:
    if today > last_updated['max'][0]:
        env_ip_df.to_sql(name='IPAM_clenup_per_env_IP_count', con=engine, if_exists = 'append', index=False, method='multi',  chunksize = 2097 // len(env_ip_df.columns))
        print("ip table updated")
    else:
        print("db did not update. there is data from today updated already..")
except:
    engine = create_engine('postgresql://cppmapi_so:dKqZpJmKz2n11Dq@postgres5561-lb-fm-in.iglb.intel.com:5433/cppmapi')
    con = engine.connect()
    env_ip_df.to_sql(name='IPAM_clenup_per_env_IP_count', con=engine, if_exists = 'append', index=False, method='multi',  chunksize = 2097 // len(env_ip_df.columns))
    print("*ip table updated")


# In[21]:


try:
    if today > last_updated['max'][0]:
        env_subnet_df.to_sql(name='IPAM_clenup_per_env_subnet_count', con=engine, if_exists = 'append', index=False, method='multi',  chunksize = 2097 // len(env_subnet_df.columns))
        print("subnet table updated")
    else:
        print("db did not update. there is data from today updated already..")
except:
    engine = create_engine('postgresql://cppmapi_so:dKqZpJmKz2n11Dq@postgres5561-lb-fm-in.iglb.intel.com:5433/cppmapi')
    con = engine.connect()
    env_subnet_df.to_sql(name='IPAM_clenup_per_env_subnet_count', con=engine, if_exists = 'append', index=False, method='multi',  chunksize = 2097 // len(env_subnet_df.columns))
    print("*subnet table updated")

