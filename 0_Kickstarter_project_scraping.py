# -*- coding: utf-8 -*-
"""

@author: Shilong

"""


# import necessary libaries to scrape and store data
import pandas as pd
import requests
import time
import random

#------------------------------------------------------------------------------
# read project data downloaded from Web Robots database
df0 = pd.read_csv('Kickstarter.csv')
df1 = pd.read_csv('Kickstarter001.csv')
df2 = pd.read_csv('Kickstarter002.csv')
df3 = pd.read_csv('Kickstarter003.csv')
df4 = pd.read_csv('Kickstarter004.csv')
df5 = pd.read_csv('Kickstarter005.csv')
df6 = pd.read_csv('Kickstarter006.csv')
df7 = pd.read_csv('Kickstarter007.csv')
df8 = pd.read_csv('Kickstarter008.csv')
df9 = pd.read_csv('Kickstarter009.csv')

# concatenate different months' data into one DataFrame
df = pd.concat([df0,df1,df2,df3,df4,df5,df6,df7,df8,df9])

# diminish some unrelated features
df_0 = df.drop(['creator','country','converted_pledged_amount',
                'currency_symbol','fx_rate','currency_trailing_code',
                'current_currency','id','photo','profile',
                'source_url','static_usd_rate','slug','location',
                'usd_type','friends','is_backing','is_starred',
                'permissions','usd_pledged','usd_type'],axis=1)

# only the kickstarter projects pleged with USD are analyzed
df_0 = df_0[df_0.currency == 'USD']
df_0.drop(['currency'],axis=1,inplace=True)

# save data into disc which may be revisited in the future
df_0.to_csv('df_0.csv',index=False)

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# scrape each project page with its URL to conduct following feature engineering
# build a new DataFrame to store scraped HTMLs
scraped_all = pd.DataFrame(columns=['scraped_info'])
request_count = 0

# scrape each project page and save response object into pickle file
for index, row in df_0.iterrows():
    scraped_piece  = requests.get(eval(row.urls)['web']['project'], timeout = 200)
  
    if index//100==0: time.sleep(random.uniform(2,4))
    
    print('Request: {}'.format(request_count))
    request_count += 1
    scraped_all.loc[index,'scraped_info'] = scraped_piece
    
scraped_all.to_pickle('scraped_all.pkl') # save scraped content into pickle file

# -----------------------------------------------------------------------------