#!/usr/bin/env python
# coding: utf-8

# In[73]:


import warnings
warnings.filterwarnings("ignore")

import pip
pckgs = ['pandas','ipaddress','numpy','tqdm','datetime','json','requests','sqlalchemy','xmltodict','re','psycopg2']
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', '--proxy=http://proxy.iil.intel.com:911', package])

for p in pckgs:
    import_or_install(p)


# In[18]:


import pandas as pd
import ipaddress as ia
from numpy import nan
from tqdm import tqdm
pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', 20)
from datetime import date
import json
import requests
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import xmltodict
import re

today = date.today()


# In[19]:


engine = create_engine('postgresql://<username>:<password>@<host>:<port>/<db_name>')
con = engine.connect()


# In[20]:


def address_exclude_collapse(net, ex_list):
    result = [net]
    for ex in ex_list:
        temp_result = []
        for sub in result:
            if ex.subnet_of(sub):
                temp_result.extend(list(sub.address_exclude(ex)))
            else:
                temp_result.append(sub)
        result = list(ia.collapse_addresses(temp_result))
    return result


# In[21]:


def is_discovered(row):
    if '*' in str(row['Last Seen']): return 'last seen old'
    elif ':' in str(row['Last Seen']): return 'in use'
    return 'never discovered'        


# In[22]:


def get_campusCode(row):
    return re.findall('([a-zA-Z ]*)\d*.*', row.BuildingCode)[0]


# In[23]:


def sort_value(row):
    parts = row.Range.split('/')[0].split('.')
    new_parts = []
    for i in range(4):
        p = parts[i]
        int_p = int(p)
        if int_p < 10:
            new_parts.append(f'00{p}')
        elif int_p < 100:
            new_parts.append(f'0{p}')
        else:
            new_parts.append(f'{p}')
    bit_part = int(row.Range.split('/')[1])
    if bit_part < 10:
        bit_part = f'00{bit_part}'
    elif bit_part < 100:
        bit_part = f'0{bit_part}'
    else:
        bit_part = str(bit_part)
        
    return f"{new_parts[0]}.{new_parts[1]}.{new_parts[2]}.{new_parts[3]}/{bit_part}"


# In[24]:


def get_children(node,childern_dict):
    if node not in childern_dict.keys():
        return []
    children = childern_dict[node]
    for c in children:
        grand_children = get_children(c,childern_dict)
        children = children + grand_children
    return children


# In[25]:


def insert_to_db(engine,table_name,df,if_exists='append'):
    chunksize = 2097 // len(df.columns)
    try:
        df.to_sql(name=table_name, con=engine, if_exists = if_exists, index=False, method='multi',  chunksize=chunksize)
    except:
        data = pd.read_sql(f'SELECT * FROM {table_name}', engine)
        df2 = pd.concat([data,df])
        chunksize2 = 2097 // len(df2.columns)
        df2.to_sql(name=table_name, con=engine, if_exists = 'replace', index=False, method='multi',  chunksize=chunksize2)


# In[26]:


def get_sixteen_summary_table(site_code='all'):   
    data = []
    for parent in sixteens:
        try:
            descendants = extended_children_dict[parent]
        except:
            descendants = [parent]
        data_dict = dict()
        data_dict['Range'] = parent
        
        if site_code == 'all':
            desc_df = new_df[(new_df.Range.isin(descendants)) & (new_df.is_leaf == True)]
        else:
            desc_df = new_df[(new_df.Range.isin(descendants)) & (new_df.is_leaf == True) & (new_df.new_siteCode == site_code)]
             
        for env in envs:
            env_df = desc_df[desc_df.new_environment == env]
            
            childern = list(env_df[env_df.is_discovered == 'never discovered'].Range)
            output_list = list(map(ia.ip_network, childern))
            data_dict[f'{env}_never_discovered'] = get_subnet_summary(output_list)

            childern = list(env_df[env_df.is_discovered == 'free'].Range)
            output_list = list(map(ia.ip_network, childern))
            data_dict[f'{env}_free'] = get_subnet_summary(output_list)

            childern = list(env_df[env_df.is_discovered == 'in use'].Range)
            output_list = list(map(ia.ip_network, childern))
            data_dict[f'{env}_in_use'] = get_subnet_summary(output_list)

            childern = list(env_df[env_df.is_discovered == 'last seen old'].Range)
            output_list = list(map(ia.ip_network, childern))
            data_dict[f'{env}_last_seen_old'] = get_subnet_summary(output_list)
        data.append(data_dict)

    summary_df_sixteen = pd.DataFrame(data=data)
    summary_df_sixteen['sort_value'] = summary_df_sixteen.apply(sort_value, axis=1)
    summary_df_sixteen = summary_df_sixteen.sort_values('sort_value').drop('sort_value', axis=1).reset_index(drop=True)
    ranges = summary_df_sixteen['Range']
    summary_df_sixteen = summary_df_sixteen[sorted(summary_df_sixteen.columns)].drop(['Range'],axis=1)
    summary_df_sixteen.insert(loc=0, column='Range', value=ranges)
    summary_df_sixteen.insert(loc=1, column='site_code', value=site_code)
    summary_df_sixteen['collection_date']  = today
    return summary_df_sixteen.loc[summary_df_sixteen.drop(['Range','collection_date','site_code'], axis=1).dropna(how='all').index]


# In[27]:


def get_subnet_summary(sub_list):
    count_dict = {}
    for sn in sub_list:
        pLen = sn.prefixlen
        if pLen in count_dict: 
            count_dict[pLen] += 1 
        else: 
            count_dict[pLen] = 1
    count_dict = {k: v for k, v in sorted(count_dict.items(), key=lambda item: item[0])}
#     if count_dict!={}:
#         print(str(count_dict))
#         raise
    if count_dict:
        return str(count_dict)
    else:
        return nan


# ### API call

# In[28]:


print('starting api call from DDI...')
url = 'http://ipam.intel.com/mmws/api/Ranges'
response = requests.get(url, proxies={'https': '', 'http': ''}, auth=(<username>, <password>),
                        verify=False,
                        timeout=100)

myjson = json.loads(response.text)
json_df = pd.DataFrame(myjson['result']['ranges'])
json_df = json_df[~json_df.name.str.contains(':')].reset_index(drop=True)


# In[29]:


# create complete dataframe with all the data collected using API
rows = []
for idx,row in tqdm(json_df.iterrows()):
    new_row = row['customProperties']
    new_row['Range'] = row['name']
    new_row['is_subnet'] = row['subnet']
    new_row['utilization'] = row['utilizationPercentage']
    rows.append(new_row)
df = pd.DataFrame(rows)
print('finished api call from DDI...')


# ### Transform

# In[30]:


df['last_seen_new'] = df['Last Seen'].apply(lambda x:  x.split(' ')[0] if (type(x) == str) else x)


# In[31]:


df['is_discovered'] = 'NA'

diff = '365'

threshold = datetime.now() - timedelta(days=int(diff))
threshold = threshold.date()
for idx,row in df.iterrows():
    if type(row.last_seen_new) == float:
        df.at[idx, 'is_discovered'] = 'never discovered'
    elif pd.to_datetime(row.last_seen_new).date() <= threshold:
        df.at[idx, 'is_discovered'] = 'last seen old'
    else:
        df.at[idx, 'is_discovered'] = 'in use'


# In[32]:


df = df.drop(['Last Seen'],axis=1).rename({'last_seen_new':'Last Seen'}, axis=1)


# In[33]:


# remove all subnets too large or too big.
# WAN team will handle the larger subnets manually and will decide whether to ignore the smaller ones completely
removables = list()
ind2net = dict() # keys are int, values are IPv4Network type
net2ind = dict() # keys are str values are int
for idx,row in df.iterrows():
    try:
        net = ia.ip_network(row.Range)
        pLen = net.prefixlen
        if not (16 <= pLen <= 29) and pLen != 0:
            removables.append(idx)
        else:
            ind2net[idx] = net
            net2ind[row.Range] = idx
    except:
        removables.append(idx)
df = df.drop(removables)


# In[34]:


# remove all subnets not under allowed /16s.

allowed = open("allowed_sixteens.txt", "r").read().split('\n')
removables = list()
df['sixteen_predecessor'] = 'none'
for idx,row in df.iterrows():
    try:
        pred = ia.ip_network(row.Range).supernet(new_prefix=16).compressed
        if pred not in allowed and row.Range != '0.0.0.0/0':
            removables.append(idx)
        else:
            df.at[idx,'sixteen_predecessor'] = pred
    except:
        removables.append(idx)
df = df.drop(removables)


# In[35]:


# decide who is the immediate parent of each node
# update childern dictionary - parent(IPv4Network): children(list of IPv4Networks)
df['parent'] = 'NA'
childern_dict = dict()
for idx,row in df.iterrows():
    if row.Range == '0.0.0.0/0':
        continue
    
    net = ind2net[idx]
    pLen = net.prefixlen
    parent = '0.0.0.0/0'
    for i in range(pLen-1,15,-1):
        supernet = net.supernet(new_prefix=i).compressed
        if supernet in net2ind:
            parent = supernet
            break
    df.at[idx,'parent'] = parent
    if parent in childern_dict: childern_dict[parent].append(net)
    else: childern_dict[parent] = [net]


# ### finding free subnets

# In[36]:


print('organizing children subnets...')
childern_dict_collapsed = {k:list(ia.collapse_addresses(v)) for (k,v) in childern_dict.items()}

print('organizing free subnets...')
free_space = {k:address_exclude_collapse(ia.ip_network(k),v) for (k,v) in childern_dict_collapsed.items()}

rows = []
for k,v in free_space.items():
    for fr in v:
        try:
            rows.append([k,'free',fr.compressed,fr.supernet(new_prefix=16).compressed])
        except:
            continue
free_df = pd.DataFrame(rows, columns=[['parent','is_discovered','Range','sixteen_predecessor']])


# In[37]:


tmp = df.head(0).copy()
tmp['Range'] = free_df['Range']
tmp['is_discovered'] = free_df['is_discovered']
tmp['parent'] = free_df['parent']
tmp['sixteen_predecessor'] = free_df['sixteen_predecessor']
new_df = pd.concat([tmp,df]).reset_index(drop=True).sort_values('Range')
new_df['sort_value'] = new_df.apply(sort_value, axis=1)
new_df['collection_date'] = today
new_df = new_df.sort_values('sort_value')[['Range','parent','sixteen_predecessor', 'is_discovered',
       'utilization', 'Title', 'Description', 'RouteAdvertised', 'SiteCode',
       'BuildingCode', 'DNSCode', 'Environment', 'Gateway', 'Vlan', 'Function',
       'Routers', 'Location', 'Status', 'Region', 'IsTopLevel',
       'SysContact', 'Netmask', 'IPHelpers', 'Country', 'SiteName',
       'Approval Group', 'Last Seen', 'SecurityRating', 'SpaceID',
       'collection_date']]
new_df = new_df[((new_df.Range.isin(allowed)) | (new_df.sixteen_predecessor.isin(allowed)) | (new_df.Range == '0.0.0.0/0'))].reset_index(drop=True)


# ### add correct environment and siteCode

# In[38]:


print("organizing environments and site codes...")
parents = set(new_df.parent) 
new_df['is_leaf'] = new_df['Range'].apply(lambda x: False if x in parents else True)


# In[39]:


childern_dict = dict()
for idx, row in new_df.iterrows():
    range_, parent_ = row.Range, row.parent
    if parent_ not in childern_dict:
        childern_dict[parent_] = [range_]
    else:
        childern_dict[parent_].append(range_)


# In[40]:


extended_children_dict = dict()
for node in childern_dict.keys():
    if type(node) == float:
        continue
    extended_children_dict[node] = get_children(node,childern_dict)


# In[41]:


# new_df['campusCode'] = new_df[new_df.BuildingCode.notna()].apply(get_campusCode, axis=1)


# In[42]:


for idx, row in tqdm(new_df[new_df.is_discovered != 'free'].fillna("none").iterrows()):
    try:
        bcs = list(set(new_df[new_df.Range.isin(extended_children_dict[row.Range])].BuildingCode.dropna()))
        bcs = [re.sub(r'\([^)]*\)', '', x).replace('*','').replace(')','').strip() for x in bcs]
        if row.BuildingCode not in bcs: bcs.append(re.findall('([a-zA-Z ]*)\d*.*', row.BuildingCode)[0])
    except:
        bcs = re.sub(r'\([^)]*\)', '', row.BuildingCode).replace('*','').replace(')','').strip()    
        
    if len(bcs) > 1:
        bc = str(bcs)
    elif len(bcs) == 1:
        bc = bcs[0]
    else:
        bc = 'none'
        
    new_df.at[idx, 'new_BuildingCode'] = bc


# In[43]:


building_df = pd.read_sql_query('select "BuildingCd", "CampusCd" from building_inventory', con=con).rename({'BuildingCd':'new_BuildingCode','CampusCd':'CampusCode'}, axis=1)


# In[44]:


new_df = new_df.merge(building_df, how='left', on='new_BuildingCode')


# In[45]:


# decide on environment and siteCode
new_df['new_environment'] = 'none'
new_df['new_siteCode'] = 'none'
new_df['new_campusCode'] = 'none'

for idx, row in tqdm(new_df[new_df.is_discovered != 'free'].fillna("none").iterrows()):
    try:
#         envs = [x.split(' ')[0] for x in list(set(new_df[new_df.Range.isin(extended_children_dict[row.Range])].Environment.dropna()))]
        envs = [re.sub("[\(\[].*?[\)\]]", "", x).strip() for x in list(set(new_df[new_df.Range.isin(extended_children_dict[row.Range])].Environment.dropna()))]
        if re.sub("[\(\[].*?[\)\]]", "", row.Environment).strip() not in envs: envs.append(re.sub("[\(\[].*?[\)\]]", "", row.Environment).strip())
    except:
        envs = re.sub("[\(\[].*?[\)\]]", "", row.Environment.split(' ')[0]).strip()
        
    try:
        scs = list(set(new_df[new_df.Range.isin(extended_children_dict[row.Range])].SiteCode.dropna()))
        scs = [re.sub("[\(\[].*?[\)\]]", "", x).strip() for x in scs]
        if re.sub("[\(\[].*?[\)\]]", "", row.SiteCode).strip() not in scs: scs.append(re.sub("[\(\[].*?[\)\]]", "", row.SiteCode).strip())
    except:
        scs = re.sub("[\(\[].*?[\)\]]", "", row.SiteCode.split(' ')[0]).strip()
        
    try:
        ccs = list(set(new_df[new_df.Range.isin(extended_children_dict[row.Range])].CampusCode.dropna()))
        if row.CampusCode not in ccs: ccs.append(row.CampusCode)
    except:
        ccs = row.CampusCode
    

    if len(envs) > 1:
        env = str(envs)
    elif len(envs) == 1:
        env = envs[0]
    else:
        env = 'none'

    if len(scs) > 1:
        sc = str(scs)
    elif len(scs) == 1:
        sc = scs[0]
    else:
        sc = 'none'
        
    if len(ccs) > 1:
        cc = str(ccs)
    elif len(ccs) == 1:
        cc = ccs[0]
    else:
        cc = 'none'

    if ',' not in env:
        env = env.split(' ')[0]
    new_df.at[idx, 'new_environment'] = env
    new_df.at[idx, 'new_siteCode'] = sc
    new_df.at[idx, 'new_campusCode'] = cc


# In[46]:


new_df.drop(['CampusCode'], axis=1, inplace=True)


# In[47]:


for idx,row in tqdm(new_df[new_df.is_discovered == 'free'].iterrows()):
    parent_data = list(new_df[new_df.Range == row.parent][['new_environment','new_siteCode','new_campusCode']].reset_index(drop=True).iloc[0])
    new_df.at[idx, 'new_environment'] = parent_data[0]
    new_df.at[idx, 'new_siteCode'] = parent_data[1]
    new_df.at[idx, 'new_campusCode'] = parent_data[2]


# In[48]:


# get top of subtrees (environment and site)
# new_df['is_top_environment'] = 'none'
# env_df = new_df[(new_df.new_environment != 'multi') & (new_df.is_discovered != 'free')]
env_df = new_df[~(new_df.new_environment.str.contains(',')) & (new_df.is_discovered != 'free')]

env_ranges = list(env_df.Range)
    
for idx,row in tqdm(env_df.iterrows()):
    if row.parent in env_ranges:
        new_df.at[idx,'is_top_environment'] = False
    else:
        new_df.at[idx,'is_top_environment'] = True
        
# new_df['is_top_siteCode'] = 'none'
# loc_df = new_df[(new_df.new_siteCode != 'multi') & (new_df.is_discovered != 'free')]
loc_df = new_df[~(new_df.new_siteCode.str.contains(',')) & (new_df.is_discovered != 'free')]

loc_ranges = list(loc_df.Range)

for idx,row in tqdm(loc_df.iterrows()):
    if row.parent in loc_ranges:
        new_df.at[idx,'is_top_siteCode'] = False
    else:
        new_df.at[idx,'is_top_siteCode'] = True
        
        
# new_df['is_top_campusCode'] = 'none'
# loc_df = new_df[(new_df.campusCode != 'multi') & (new_df.is_discovered != 'free')]
camp_df = new_df[~(new_df.new_campusCode.str.contains(',')) & (new_df.is_discovered != 'free')]

camp_ranges = list(camp_df.Range)

for idx,row in tqdm(camp_df.iterrows()):
    if row.parent in camp_ranges:
        new_df.at[idx,'is_top_campusCode'] = False
    else:
        new_df.at[idx,'is_top_campusCode'] = True


# In[49]:


new_df.fillna({'is_top_siteCode':False, 'is_top_environment':False,'is_top_campusCode':False}, inplace=True)


# In[50]:


new_df


# ### Data Storage

# In[51]:


# # CSV local backup
# new_df.to_csv("ordered_subnets.csv", index = False)


# In[52]:


# new_df = pd.read_csv('ordered_subnets.csv')


# In[53]:


# SQL db insertion
print("inserting to db...")
new_df.to_sql(name='full_ipam_fragmentation', con=engine, if_exists = 'replace', index=False, method='multi',  chunksize = 2097 // len(new_df.columns))
print("db updated with IPAM transformed data...")


# In[54]:


new_df.to_csv("ordered_subnets.csv", index = False)


# ### allowed /16 subnet handling

# In[55]:


print("handling /16 subnets' summaries...")
sixteens = list(new_df[(new_df.Range.str.contains("/16")) & (new_df.is_discovered != 'free')].Range)
envs = list(list(new_df[~new_df.new_environment.str.contains(',')].new_environment.fillna('none').unique()))
scs = ["all"] + list(list(new_df[~new_df.new_campusCode.str.contains(',')].new_campusCode.fillna('none').unique()))


# In[56]:


data_frames = []
for sc in tqdm(scs):
    data_frames.append(get_sixteen_summary_table(sc))


# In[57]:


total_sixteen = pd.concat(data_frames)
total_sixteen = total_sixteen.dropna(how='all', axis=1)


# In[58]:


# total_sixteen.to_csv('subnet_tree_summary_sixteens.csv', index=False)


# In[75]:


total_sixteen


# In[59]:


try:
    total_sixteen.to_sql(name='sixteens_summary', con=engine, if_exists = 'replace', index=False, method='multi',  chunksize = 2097 // len(total_sixteen.columns))
except:
    engine = create_engine('postgresql://cppmapi_so:dKqZpJmKz2n11Dq@postgres5561-lb-fm-in.iglb.intel.com:5433/cppmapi')
    total_sixteen.to_sql(name='sixteens_summary', con=engine, if_exists = 'replace', index=False, method='multi',  chunksize = 2097 // len(total_sixteen.columns))


# ### top environment\siteCode\campusCode trees handling

# In[60]:


# create heritage table
rows = []
for pred,succ_list in extended_children_dict.items():
    if pred == 'NA' or pred == '0.0.0.0/0':
        continue
    for succ in succ_list:
        rows.append({"predecessor":pred, "successor": succ})
heritage_df = pd.DataFrame(rows)


# In[61]:


try:
    heritage_df.to_sql(name='heritage_table', con=engine, if_exists = 'replace', index=False, method='multi',  chunksize = 2097 // len(heritage_df.columns))
except:
    engine = create_engine('postgresql://<username>:<password>@<host>:<port>/<db_name>')
    heritage_df.to_sql(name='heritage_table', con=engine, if_exists = 'replace', index=False, method='multi',  chunksize = 2097 // len(heritage_df.columns))


# In[62]:


engine = create_engine('postgresql://<username>:<password>@<host>:<port>/<db_name>')
con = engine.connect()
data_orig = pd.read_sql_query('select * from full_ipam_fragmentation where ("is_top_campusCode") or ("is_top_environment") or ("is_top_siteCode")', con=con)
heritage_table = pd.read_sql_query('select * from heritage_table', con=con)


# In[63]:


filtered_df = data_orig.copy()
ranges = list(filtered_df.Range)
data = []


# In[64]:


ranges = list(filtered_df.Range)
data = []
for parent in tqdm(ranges):
    try:
        descendants = list(heritage_table[heritage_table.predecessor == parent].successor)
        if not descendants:
            descendants = [parent]
    except:
        descendants = [parent]
    data_dict = dict()
    data_dict['Range'] = parent


    desc_df = filtered_df[(filtered_df.Range.isin(descendants)) & (filtered_df.is_leaf == True)]

    for env in envs:
        env_df = desc_df[desc_df.new_environment == env]

        childern = list(env_df[env_df.is_discovered == 'never discovered'].Range)
        output_list = list(map(ia.ip_network, childern))
        data_dict[f'{env}_never_discovered'] = get_subnet_summary(output_list)

        childern = list(env_df[env_df.is_discovered == 'free'].Range)
        output_list = list(map(ia.ip_network, childern))
        data_dict[f'{env}_free'] = get_subnet_summary(output_list)

        childern = list(env_df[env_df.is_discovered == 'in use'].Range)
        output_list = list(map(ia.ip_network, childern))
        data_dict[f'{env}_in_use'] = get_subnet_summary(output_list)

        childern = list(env_df[env_df.is_discovered == 'last seen old'].Range)
        output_list = list(map(ia.ip_network, childern))
        data_dict[f'{env}_last_seen_old'] = get_subnet_summary(output_list)
    data.append(data_dict)


# In[65]:


summary_df = pd.DataFrame(data=data)
summary_df['sort_value'] = summary_df.apply(sort_value, axis=1)
summary_df = summary_df.sort_values('sort_value').drop('sort_value', axis=1).reset_index(drop=True)
ranges = summary_df['Range']
summary_df = summary_df[sorted(summary_df.columns)].drop(['Range'],axis=1)
summary_df.insert(loc=0, column='Range', value=ranges)
summary_df.insert(loc=1, column='site_code', value="all")
summary_df['collection_date']  = today
summary_df = summary_df.loc[summary_df.drop(['Range','collection_date','site_code'], axis=1).dropna(how='all').index]


# In[66]:


summary_df = summary_df[[col for col in summary_df.columns if ',' not in col]].dropna(how='all', axis=1)


# In[67]:


summary_df.to_csv("summary_data.csv", index=False)


# In[68]:


aux_df = new_df[['Range','new_environment','new_siteCode','new_campusCode','is_top_siteCode','is_top_environment','is_top_campusCode']].copy()


# In[69]:


summary_final = summary_df.merge(aux_df, left_on='Range', right_on='Range', how='left').drop('site_code', axis=1)


# In[70]:


top_cols = ['Range','new_environment', 'new_siteCode', 'new_campusCode']
bttm_cols = ['is_top_environment','is_top_siteCode','is_top_campusCode','collection_date']
cols = top_cols + sorted([col for col in summary_final.columns if col not in top_cols + bttm_cols]) + bttm_cols
summary_final = summary_final[cols]


# In[71]:


summary_final


# In[72]:


try:
    summary_final.to_sql(name='top_summary', con=engine, if_exists = 'replace', index=False, method='multi',  chunksize = 2097 // len(summary_final.columns))
except:
    engine = create_engine('postgresql://<username>:<password>@<host>:<port>/<db_name>')
    summary_final.to_sql(name='top_summary', con=engine, if_exists = 'replace', index=False, method='multi',  chunksize = 2097 // len(summary_final.columns))


# In[74]:


summary_final
