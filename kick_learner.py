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
# ### In this project, we use the Kickstarter Projects dataset to predict whether a campaign will be successful or not based on features such as ...
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
# df = pd.read_csv("ks-projects-201612.csv", low_memory = False)
df = pd.read_csv("ks-projects-201612.csv", encoding="cp1252", low_memory=False) # to make encoding work, at elast on macOS

# %% jupyter={"outputs_hidden": true}
df.columns = df.columns.str.strip()
df.sample(5)

# %% jupyter={"outputs_hidden": true}
df.info()

# %% [markdown]
# ## Data Exploration

# %%
df.columns

# %%
df['state'].value_counts()

# %%
df[df['state'] == "canceled"].head()

# %% jupyter={"outputs_hidden": true}
dftest = df[df['country'] == 'US']
dftest = dftest.drop(columns=["Unnamed: 13", "Unnamed: 14", "Unnamed: 15", "Unnamed: 16"])

# %%
dftest['category'].value_counts()

# %%
dftest['main_category'].value_counts()

# %% [markdown]
# #### We want to see the different outcomes the campaigns had, and which ones will be most relevant for our predictions. 

# %%
state_counts = df['state'].value_counts().head(10)
state_counts.plot.bar()
plt.title('Distribution of campaign outcomes')
plt.xlabel('state')
plt.ylabel('count')
plt.show()

# %% [markdown]
# #### Here we are able to see that "Failed" and "Successful" are the most common. We want to explore if "canceled" should fall under the "failed" category.

# %% [markdown]
# #### We want to see how Kickstarter campaigns are distributed across different project categories. This will help us understand which categories are most popular and whether there is a class imbalance. 

# %%
df['main_category '].value_counts().head(10).plot.bar()
plt.title("Number of campaigns per main category")
plt.xlabel("main category")
plt.ylabel("count")
plt.show()

# %%

# %% jupyter={"outputs_hidden": true}
df.sample(5)

# %% [markdown]
# ## Preprocessing /Data Cleaning

# %%
df = df.drop(columns=["Unnamed: 13", "Unnamed: 14", "Unnamed: 15", "Unnamed: 16"])

# %% jupyter={"outputs_hidden": true}
df.sample(5)

# %% jupyter={"outputs_hidden": true}
df[df.isnull().any(axis=1)].sample(5)

# %%
df = df.rename(columns={"state ": "state"}) # There's a trailing space in the column name that is annoying
df = df[df['state'].isin(["successful", "failed	", "canceled"])]

# %% jupyter={"outputs_hidden": true}
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

# %%
df['backers'].info()

# %%
df['goal'] = pd.to_numeric(df['goal'], errors='coerce').astype(int)
df.dropna(subset=['goal'], inplace=True)
print(df['goal'])

# %%
#Dropping all rows outside the US
df = df[df['country'] == 'US']
