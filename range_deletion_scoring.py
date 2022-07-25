#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sqlalchemy import create_engine
import ipaddress as ia
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings("ignore")
from py_topping.data_connection.sharepoint import lazy_SP365


# In[2]:


# Import library
# Create connection
sp = lazy_SP365(site_url = 'https://intel.sharepoint.com/sites/IPAMDataQuality'
                   , client_id = 'd41c4361-de53-4b02-a260-516a66728c62'
                   , client_secret = 'ZFxlzvWnWvAC7to9irYF0k7RKnjmSNEVC8W/da37RuQ=')
# Create download path from download URL
download_path = sp.create_link('https://intel.sharepoint.com/:x:/r/sites/IPAMDataQuality/Shared%20Documents/General/2022_IPAM_range_mgmt_data_quaility/prioritized_ranges/prioritized_ranges.xlsx?d=w50030ee0da31460eb0db753ad7d91a32&csf=1&web=1&e=lS0n7o')
# Download file
sp.download(sharepoint_location = download_path, local_location = rf"prioritized_ranges.xlsx")
prioritized = pd.read_excel("prioritized_ranges.xlsx")


# In[3]:


prioritized


# In[4]:


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


# In[5]:


def sort_value_var(ip):
    parts = ip.split('.')
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

    return f"{new_parts[0]}.{new_parts[1]}.{new_parts[2]}.{new_parts[3]}"


# In[6]:


def fix_env_string(val):
    try:
        return val if "," in val else eval(val)[0]
    except:
        return val


# In[7]:


def create_scoring_dict(df, env, score_dict):
    # seperating indexes - not relevant
    if env == 'all':
        idx_list = list(df[df.is_discovered.isin(["in use", "free","KEEP"]) | (df.Range.str.contains('/16'))].index)
    else:
        idx_list = list(df[(df.is_discovered.isin(["in use", "free","KEEP"])) | (df.Range.str.contains('/16')) | (
                    df.new_environment != env)].index)

    dfs = [df.iloc[0:idx_list[0]]]
    if dfs[0].empty:
        dfs = []

    for i in range(df.shape[0]):
        try:
            from_, to_ = idx_list[i] + 1, idx_list[i + 1]
        except:
            break
        if from_ == to_:
            continue
        temp_df = df.iloc[from_:to_]
        if not temp_df.empty:
            dfs.append(temp_df.reset_index(drop=True))
    data = []
    for df_ in tqdm(dfs):
        nunique = df_.nunique()
        sum_df = df_.sum()
        row = {"first": df_.iloc[0]["first"], "last": df_.iloc[-1]["last"], "ip_num": sum_df["ip_num"],
               "subnet_num": sum_df["subnet_num"], "unique_env_num": nunique["new_environment"],
               "environments": str(list(df_.new_environment.unique())) if len(df_.new_environment.unique()) > 1 else list(df_.new_environment.unique())[0]}
        data.append(row)
    new_df = pd.DataFrame(data=data)

    # TODO: think if there is any better scoring method. For now avg subnet size.
    new_df["avg_subnet_size"] = new_df.ip_num // new_df.subnet_num
    new_df = new_df.sort_values("avg_subnet_size", ascending=False)
    score_dict[env] = new_df


# In[8]:


engine = create_engine('postgresql://cppmapi_so:dKqZpJmKz2n11Dq@postgres5561-lb-fm-in.iglb.intel.com:5433/cppmapi')
con = engine.connect()
df = pd.read_sql(rf"""select "Range","is_discovered", "new_environment", "sixteen_predecessor" from public."IPAM_full_fragmentation"
where (is_leaf)""", engine)


# In[9]:


df['first'] = df['Range'].apply(lambda x: str(ia.IPv4Network(x)[0]))
df['last'] = df['Range'].apply(lambda x: str(ia.IPv4Network(x)[-1]))
df['ip_num'] = df['Range'].apply(lambda x: ia.IPv4Network(x).num_addresses)
df['subnet_num'] = 1
df['sort_value'] = df.apply(sort_value, axis=1)


# In[10]:


df = df.sort_values("sort_value").reset_index(drop=True)


# In[11]:


envs = ['all'] + list(df[(~df.is_discovered.isin(["in use", "free","KEEP"])) & (~df.new_environment.isin(['none']))].new_environment.unique())


# In[12]:


# dfs = [df.iloc[0:idx_list[0]]]
# for i in range(df.shape[0]):
#     try:
#         from_, to_ = idx_list[i]+1,idx_list[i+1]
#     except:
#         break
#     if from_ == to_:
#         continue
#     temp_df = df.iloc[from_:to_]
#     dfs.append(temp_df.reset_index(drop=True))


# In[13]:


# data = []
# for df_ in tqdm(dfs):
#     nunique = df_.nunique()
#     sum_df = df_.sum()
#     row = {"first":df_.iloc[0]["first"], "last":df_.iloc[-1]["last"], "ip_num":sum_df["ip_num"],
#            "subnet_num":sum_df["subnet_num"], "unique_env_num":nunique["new_environment"],
#            "unique_site_num":nunique["new_siteCode"],
#            "unique_campus_num":nunique["new_campusCode"],
#            "unique_building_num":nunique["new_BuildingCode"]}
#     data.append(row)
# new_df = pd.DataFrame(data=data)


# In[14]:


# new_df["avg_subnet_size"] = new_df.ip_num // new_df.unique_env_num


# In[15]:


# new_df = new_df.sort_values("unique_env_num", ascending=False)
# new_df


# In[16]:


score_dict = dict()
for env in envs:
    print(env)
    create_scoring_dict(df, env, score_dict)


# In[17]:


# with open('score_dict.pickle', 'rb') as handle:
#     score_dict = pickle.load(handle)


# In[18]:


for k in score_dict:
    score_dict[k]['environments'] = score_dict[k]['environments'].apply(fix_env_string)


# In[19]:


score_dict['all'] = score_dict['all'].sort_values(['avg_subnet_size', 'unique_env_num'], ascending=[False, True]).reset_index(drop=True).reset_index()


# In[20]:


df['first_sort_value'] = df['first'].apply(sort_value_var)


# In[21]:


dfs = []
for env_input in envs[1:] + ['none']:
    print(env_input)
    relevant_df = score_dict['all'][score_dict['all'].environments.str.contains(env_input)]
    rel_subnets = []
    for i in tqdm(range(relevant_df.shape[0])):
        rows = df[(df.new_environment == env_input) & (df.first_sort_value.between(sort_value_var(relevant_df.iloc[i]['first']),sort_value_var(relevant_df.iloc[i]['last'])))]
        rows['macro_index'] = relevant_df.iloc[i]['index']
        rel_subnets.append(rows)
    final_df = pd.concat(rel_subnets).drop(['sort_value','first_sort_value','subnet_num'], axis=1).reset_index(drop=True).reset_index()
    if 'Unnamed: 0' in final_df.columns:
        final_df.drop('Unnamed: 0', axis=1, inplace=True)
    final_df = final_df[["macro_index"] + list(final_df.columns[:-1])]
    dfs.append(final_df)


# In[22]:


final_con = pd.concat(dfs).sort_values(["macro_index","index"]).reset_index(drop=True).rename({"index":"index_per_APT"}, axis=1)


# In[23]:


final_con_pr = pd.concat([final_con[final_con.sixteen_predecessor.isin(prioritized.Range)],final_con[~final_con.sixteen_predecessor.isin(prioritized.Range)]])


# In[24]:


final_con_pr = final_con_pr.reset_index(drop=True).reset_index().rename({"index":"priority_index"},axis=1)
final_con_pr


# In[25]:


try:
    final_con_pr.to_sql(name='IPAM_range_deletion_candidates', con=engine, if_exists = 'replace', index=False, method='multi',  chunksize = 2097 // len(final_con_pr.columns))
except:
    engine = create_engine('postgresql://cppmapi_so:dKqZpJmKz2n11Dq@postgres5561-lb-fm-in.iglb.intel.com:5433/cppmapi')
    final_con_pr.to_sql(name='IPAM_range_deletion_candidates', con=engine, if_exists = 'replace', index=False, method='multi',  chunksize = 2097 // len(final_con_pr.columns))


# In[26]:


con.close()

