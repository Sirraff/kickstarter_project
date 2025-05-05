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
df = pd.read_csv("ks-projects-201612.csv", low_memory = False)

# %%
df.head()

# %%
df.info()

# %% [markdown]
# ## Data Exploration

# %%
df.dropna()
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

# %% [markdown]
# ## Preprocessing

# %% [markdown]
# ## Test/Train Split

# %% [markdown]
# ## Baseline Accuracy

# %%
