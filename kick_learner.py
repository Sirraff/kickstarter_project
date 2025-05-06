# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
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

# %%
df.sample(10)

# %%
df.info()

# %% [markdown]
# ## Data Exploration

# %%
df.columns

# %%
df['state '].value_counts()

# %%
df[df['state '] == "canceled"].head()

# %%
state_counts = df['state'].value_counts().head(10)
state_counts.plot.bar()
plt.title('Distribution of campaign outcomes')
plt.xlabel('state')
plt.ylabel('count')
plt.show()

# %%
df.sample(10)

# %% [markdown]
# ## Preprocessing /Data Cleaning

# %%
df = df.drop(columns=["Unnamed: 13", "Unnamed: 14", "Unnamed: 15", "Unnamed: 16"])

# %%
df.sample(10)

# %%
df[df.isnull().any(axis=1)].sample(10)

# %%
df = df.rename(columns={"state ": "state"}) # There's a trailing space in the column name that is annoying
df = df[df['state'].isin(["successful", "failed	", "canceled"])]

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

# %%

# %%
