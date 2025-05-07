# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %% [markdown]
# ## CST 383: Intro to Data Science
# # Project 2 
#
# # Predicting Kickstarter Campaign Success
# ## Authors: Brianna Magallon, Tyler Pruitt, Rafael Reis

# %% [markdown]
# # Introduction: 
# #### In this project, we use the Kickstarter Projects dataset to predict whether a campaign will be successful or not. The dataset includes campaign data like goal amount, category, duration, and currency. Our goal is to predict campaign success using classification models. 
# ### Dataset URL: https://www.kaggle.com/datasets/kemical/kickstarter-projects

# %%
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

# %%
# plotting
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.figsize'] = 5,3

# %% [markdown]
# ## Read the data

# %%
df = pd.read_csv("ks-projects-201612.csv", low_memory = False)
#df = pd.read_csv("ks-projects-201612.csv", encoding="cp1252", low_memory=False) # to make encoding work, at elast on macOS

# %%
#remove trailing spaces
df.columns = df.columns.str.strip()

# %%
df.sample(5)

# %%
df.info()

# %% [markdown]
# ## Data Exploration

# %% [markdown]
# #### In this section, we want to explore key aspects of the kickstarter data, including campaign outcomes, category distribution, and common funding goals. This will help us understand potential predictors of success. 

# %% [markdown]
# A look at the columns.

# %%
df.columns

# %%
df[df['state'] == "canceled"].head()

# %%
dftest = df[df['country'] == 'US']
dftest = dftest.drop(columns=["Unnamed: 13", "Unnamed: 14", "Unnamed: 15", "Unnamed: 16"])

# %%
dftest['category'].value_counts()

# %%
dftest['main_category'].value_counts()

# %% [markdown]
# We want to see the different outcomes the campaigns had, and which ones will be most relevant for our predictions. 

# %%
state_counts = df['state'].value_counts().head(10)
state_counts.plot.bar()
plt.title('Distribution of campaign outcomes')
plt.xlabel('state')
plt.ylabel('count')
plt.show()

# %% [markdown]
# Here we are able to see that "Failed" and "Successful" are the most common. Most campaings fail and there is an imbalance. 

# %% [markdown]
# Here is the exact number in each of these states.

# %%
df['state'].value_counts()

# %% [markdown]
# We want to see how Kickstarter campaigns are distributed across different project categories. This will help us understand which categories are most popular and whether there is an imbalance. 

# %%
df['main_category'].value_counts().head(10).plot.bar()
plt.title("Number of campaigns per main category")
plt.xlabel("main category")
plt.ylabel("count")
plt.show()

# %% [markdown]
# This plot shows us that Film & Video, Music, Publishing are the three most popular Kickstarter categories, while Food, Fashion and Theater are the three least common. 

# %% [markdown]
# Let’s explore which goal values appear most often. This will help us identify any odd entries or common default values.

# %%
df['goal'].value_counts().head(10).plot.bar()
plt.title("Most common goal amounts")
plt.xlabel("goal")
plt.ylabel("count")
plt.show()

# %% [markdown]
# We are able to see that the most common goal amounts are round numbers like 5000, 1000, and 10000.

# %% [markdown]
# ## Preprocessing /Data Cleaning

# %% [markdown]
# #### Before applying any machine learning models, we need to clean and prepare the data. 

# %% [markdown]
# Here we are removing irrelevant columns that contain no useful information. 

# %%
df = df.drop(columns=["Unnamed: 13", "Unnamed: 14", "Unnamed: 15", "Unnamed: 16"])

# %%
df[df.isnull().any(axis=1)].sample(5)

# %% [markdown]
# Keep only rows where 'state' is one of the target outcomes

# %%
df = df[df['state'].isin(["successful", "failed", "canceled"])]

# %%
print("Rows still with null values: ", len(df[df.isnull().any(axis=1)]))
df[df.isnull().any(axis=1)].sample(5)

# %% [markdown]
# It seems the vast majority of Kickstarter campaigns with null values fall under the Music and Film categories, often with zero backers. The campaign states vary, which likely reflects that creating music or films isn’t strongly tied to financial backing (there’s probably a joke in there somewhere).
#
# Since these projects lack key information like backer count or country and only account for 127 rows, we’ll remove them from the dataset.

# %%
df = df.dropna()
len(df[df.isnull().any(axis=1)]) # checking
df.info() # Current state

# %% [markdown]
# Convert 'goal' column to numeric, remove any rows where conversion fails

# %%
df['goal'] = pd.to_numeric(df['goal'], errors='coerce').astype(int)
df.dropna(subset=['goal'], inplace=True)
print(df['goal'])

# %% [markdown]
# Keep only US campaigns

# %%
#Dropping all rows outside the US
df = df[df['country'] == 'US']
