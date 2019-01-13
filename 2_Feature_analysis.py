# -*- coding: utf-8 -*-
"""
@author: Shilong

"""
# data cleaning and feature analysis

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------------------------------
# preliminary data cleaning
# read raw meta features data files
df = pd.read_pickle("raw_meta_feature.pkl")

# extract the category information for each project
df.category = df.category.apply(lambda x: eval(x)['name'])
      
# only analyze successful and failed projects by deleting 'live', 'canceled' and 'suspended' projects
df = df[(df.state != 'live') & (df.state != 'canceled') & (df.state != 'suspended')]

df['duration'] = df['deadline']- df['launched_at']
# fill NaN with zeros

df.fillna(0, inplace = True)

# delete unnecessary features (columns)
drop_columns = ['backers_count', 'blurb', 'created_at', 'deadline', 'disable_communication', 'is_starrable', 
                'launched_at', 'name', 'spotlight', 'staff_pick', 'state_changed_at', 'urls']

df.drop(columns = drop_columns, inplace = True)
# ----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# feature analysis 

# target balance analysis
cnt_srs = df.state.value_counts()
plt.figure(figsize = (8, 4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha = 0.8)
plt.ylabel('Number', fontsize=12)
plt.xlabel('State', fontsize=12)
plt.show()

# outliner detection
sns.distplot(df.pledged[df.pledged<10000])
sns.distplot(df.pledged[(df.pledged>10000) & (df.pledged<100000)])
sns.distplot(df.pledged[(df.pledged>100000) & (df.pledged<1000000)])
sns.distplot(df.pledged[df.pledged>1000000])
# delete outliner
df = df[df.pledged != df.pledged.max()]

# transfer categorical label type into numerical
state_mapping_dict = {'successful': 1, 'failed': 0}
df['state'] = df['state'].map(state_mapping_dict)

# correlation analysis
def plot_correlation_map( df ):
    """Plot correlation map
    
    Arg:
        df (DataFrame): DataFrame to plot correlation map
    Returns:
        No returns, plot the figure"""
        
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, 
                    ax=ax, annot = True, annot_kws = { 'fontsize' : 8 })
    
# plot correlaiton matrix 
plot_correlation_map(df)


# feature correlations between successful and failed projects
def plot_distribution( df , var , target , **kwargs ):
    """Plot kernel density estimation distributions
    
    Args:
        df (DataFrame): DataFrame to be used to plot
        var (str): string representing the column to be plotted
        target (str): string representing the target column
        kwargs: addition parameters
    Returns:
        No return.  plot distribution figures"""
        
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=1 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True,  ).fig.subplots_adjust(wspace=0.05, hspace = 0.05)
    facet.set(ylabel = 'PDF')
    facet.add_legend()
    new_labels = ['failed', 'successful']
    for t, l in zip(facet._legend.texts, new_labels): t.set_text(l)

for var in df.columns[4:20]:
    plot_distribution(df, var = var, target = 'state')

plot_distribution(df, var = 'duration', target = 'state')


# category analysis
df_2 = df.category.str.get_dummies()
category_count = df_2.sum()

for col in df_2.columns:
    df_2[col] = (df.state) & (df_2[col])

category_succ_count = df_2.sum()

df_3 = pd.concat([category_count,category_succ_count],axis=1)  
df_3[2] = df_3[1]/df_3[0]

f, ax = plt.subplots(figsize=(10, 5))
plt.xticks(rotation='vertical')
ax = plt.bar(np.array(df_3.index),df_3[2].values)

# 
df.drop(columns=['avg_words_per_sent','category','num_videos','num_gifs','duration'],inplace=True)
df = pd.concat([df,df_2],axis=1)

df.to_pickle('Processed_features.pkl')

# -----------------------------------------------------------------------------


