# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:35:20 2022

@author: ashwi
"""
import pandas as pd 

df =pd.read_csv("book.csv", encoding='latin-1')
df
df.shape
df.dtypes
df.head()

df = df.drop(df.columns[0],axis=1)

df.dtypes

df.shape
df["User.ID"].duplicated()
(df["User.ID"].duplicated()).sum()

df.sort_values("User.ID")

# number of unique users in the dataset
df["User.ID"].unique()
len(df["User.ID"].unique())

df["Book.Rating"].value_counts()
df["Book.Rating"].hist()

df["Book.Title"].value_counts()
len(df["Book.Title"].value_counts())
# 2182 users are reading 9659 books


user_df = df.pivot_table(index="User.ID",columns="Book.Title",values="Book.Rating")
pd.set_option("display.max_columns",9659)
user_df

# exported the file to excel to visualize the data
user_df.to_csv("user_df.csv")

#impute those NaNs with 0 values
user_df.fillna(0,inplace=True)
user_df

# exported the file to excel to visualize the data
user_df.to_csv("user_df1.csv")

# Applying cosine-based similarity
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances(user_df.values,metric="cosine")

user_sim

# store the results in DataFrame
user_sim_df = pd.DataFrame(user_sim)
user_sim_df

# set the index & cloumns names to user.IDs
user_sim_df.index = df['User.ID'].unique()
user_sim_df.columns = df['User.ID'].unique()

user_sim_df

user_sim_df.iloc[0:5,0:5]

# change the diagonal values to 0 values
import numpy as np
np.fill_diagonal(user_sim,0)
user_sim_df

user_sim_df.max()

# finding which two people cosine similarity is there
user_sim_df.idxmax(axis=1)[0:100]

df[(df["User.ID"]==276729) | (df["User.ID"]==276726)]
df[(df["User.ID"]==276736) | (df["User.ID"]==276726)]
df[(df["User.ID"]==276748) | (df["User.ID"]==161677)]













