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
# # CST 383 – Intro to Data Science
# ### Project 2: Predicting Kickstarter Goal Completion
# **Authors:** Brianna Magallon, Tyler Pruitt, Rafael L.S. Reis

# %% [markdown]
# ## Introduction
# In this project, we use the Kickstarter Projects dataset to build a model that predicts whether a crowdfunding campaign will succeed or fail based on information available at launch. Each entry includes metadata such as goal amount, number of backers, campaign duration, and category.
#
# We treat this as a binary classification problem, where the outcomes are `'successful'` or `'failed'`. We merge `'canceled'` campaigns into the `'failed'` category, based on the observation that they typically don't meet funding goals.
#
# **Dataset Source:**  
# [Kickstarter Projects (Kaggle)](https://www.kaggle.com/datasets/kemical/kickstarter-projects)

# %%
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# %%
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.figsize'] = 5,3

# %% [markdown]
# ## Data Exploration

# %%
df = pd.read_csv("ks-projects-201612.csv", encoding="cp1252", low_memory=False)

# %%
df.columns = df.columns.str.strip()
df.sample(5)

# %%
df.info()

# %%
df['state'].value_counts()

# %%
df[df['state'] == "canceled"].head()

# %%
df['country'].value_counts()

# %%
dftest = df[df['country'] == 'US']
dftest = dftest.drop(columns=["Unnamed: 13", "Unnamed: 14", "Unnamed: 15", "Unnamed: 16"])

# %%
category_counts = dftest['category'].value_counts().head(15)
category_counts.plot.barh()
plt.title('Distribution of categories')
plt.xlabel('count')
plt.ylabel('category')
plt.show()

# %%
main_category_count = dftest['main_category'].value_counts()
main_category_count.plot.barh()
plt.title('Distribution of main categories')
plt.xlabel('count')
plt.ylabel('category')
plt.show()

# %%
state_counts = df['state'].value_counts().head(10)
state_counts.plot.bar()
plt.title('Distribution of campaign outcomes')
plt.xlabel('state')
plt.ylabel('count')
plt.show()

# %%
df['main_category'].value_counts().head(10).plot.bar()
plt.title("Number of campaigns per main category")
plt.xlabel("main category ")
plt.ylabel("count")
plt.show()

# %% [markdown]
# ## Data Cleaning & Preprocessing
#
# We begin cleaning by dropping empty or irrelevant columns and filtering to U.S.-based projects.

# %%
df = df.drop(columns=["Unnamed: 13", "Unnamed: 14", "Unnamed: 15", "Unnamed: 16"])

# %%
df[df.isnull().any(axis=1)].sample(5)

# %%
df = df[df['state'].isin(["successful", "failed", "canceled"])]

# %%
print("Rows still with null values: ", len(df[df.isnull().any(axis=1)]))
df[df.isnull().any(axis=1)].sample(5)

# %% [markdown]
# The majority of rows with null values are music or film projects with 0 backers and questionable labels. There's probably a starving artist joke somewhere in there. These rows are minimal (~127), so we drop them.

# %%
df = df.dropna()
len(df[df.isnull().any(axis=1)])  # confirm
df.info()

# %%
df['backers'].info()

# %% [markdown]
# We now correct numeric fields and focus on U.S. projects. Some columns have numeric values stored as strings, so we convert them. This helps avoid bugs later.

# %%
df['goal'] = pd.to_numeric(df['goal'], errors='coerce').astype(int)
df.dropna(subset=['goal'], inplace=True)
print(df['goal'])

# %%
df = df[df['country'] == 'US']
df = df.drop(columns=["usd pledged"])

# %% [markdown]
# We convert string dates to datetime, compute duration, and encode labels for our classification model. We also merge `canceled` with `failed`, since both don't meet funding goals — canceled ones just end early.

# %%
df['launched'] = pd.to_datetime(df['launched'])
df['deadline'] = pd.to_datetime(df['deadline'])
df['duration_days'] = (df['deadline'] - df['launched']).dt.days

df['state'] = df['state'].replace('canceled', 'failed')
df = df[df['state'].isin(['failed', 'successful'])].copy()
df['state_encoded'] = df['state'].map({'failed': 1, 'successful': 0})

numeric_cols = ['goal', 'pledged', 'backers', 'pledged']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['launch_day'] = df['launched'].dt.day
df['launch_month'] = df['launched'].dt.month

df = df.drop(columns=['ID', 'launched', 'deadline', 'state', 'country', 'currency', 'pledged'])

# %% [markdown]
# We are dropping these columns for a variey of reasons. In the case of ID, country, and currency. The contained information is not necessary for predictions as we are focusing on projects in the US, which all use the same currency. We are replacing state with "state_encoded" making the original unecessary. Finally, pledged isn't helpful for the sake of prediction due to the fact that it directly tells the end result, making it useless for the sake of predicting the result.

# %%
df.info()

# %%
df.sample(5)

# %% [markdown]
# ## Machine Learning
#
# We define our features and target, then apply a baseline and two models. We'll compare their performance to understand how well basic models do on this problem.

# %%
X = df[['goal', 'backers', 'duration_days','launch_day','launch_month']]
y = df['state_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)
print("Baseline accuracy:", accuracy_score(y_test, y_pred_baseline))

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# %% [markdown]
# Both models beat the baseline, but KNN appears to be overfitting. More work is needed to improve generalization.

# %%
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test)
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
plt.title("KNN - Confusion Matrix")
plt.show()

# %%
coeffs = pd.Series(logreg.coef_[0], index=X.columns)
coeffs.sort_values().plot(kind='barh')
plt.title("Logistic Regression Feature Coefficients")
plt.xlabel("Impact on 'Failure' Probability")
plt.show()

# %% [markdown]
# The feature coefficients are helpful for interpretation. Backers and pledged amount seem to be the strongest predictors.

# %%
df.drop(columns=['backers']).sample(5)

# %%

# %% [markdown]
# ## Conclusion
#
# This project demonstrates the potential of using simple machine learning models to predict Kickstarter campaign outcomes based on basic project metadata. While we achieved accuracy improvements over a baseline dummy classifier, the models are still far from production-ready.
#
# We found that:
# - **Logistic Regression** provided reasonable performance and interpretable coefficients, highlighting the importance of features like number of backers and pledged amount.
# - **K-Nearest Neighbors** appeared to overfit the training data, performing well on training but less effectively on the test set.
# - Both models, while better than guessing the majority class, still left substantial room for improvement.
#
# ### Challenges & Next Steps:
# - The overfitting observed, especially in KNN, suggests a need for perhaps **feature scaling**, **regularization**, or **simplification**.
# - We haven’t yet made use of potentially informative categorical features like main_category or category.
# - Adding **cross-validation**, **feature engineering**, and testing other models (e.g., decision trees, gradient boosting) would be valuable next steps.
#
# While this was a solid start, there’s still a lot of work ahead to build a robust and generalizable model. That said, we're excited to dive into model tuning and optimization since this is the part we all find most engaging. The data cleaning process was a bit frustrating and a bit of a pain sometimes... so we're glad to have made it through that and can now focus on the fun side of machine learning :)
#
#
#
# (I know we might do some more cleaning stuff, but we're mostly done with that)
