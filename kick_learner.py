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
# # Predicting Kickstarter Goal Completion
# **Authors:** Brianna Magallon, Tyler Pruitt, Rafael L.S. Reis

# %% [markdown]
# ## Introduction
# In this project, we use the Kickstarter Projects dataset to build a model that predicts whether a crowdfunding campaign will succeed or fail based on information available at launch. This helps creators set realistic goals and improve campaign design. Each entry includes metadata such as goal amount, number of backers, campaign duration, and category.
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# %%
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.figsize'] = 5,3

# %% [markdown]
# ## Data Exploration

# %%
df = pd.read_csv("ks-projects-201612.csv", low_memory=False)
#df = pd.read_csv("ks-projects-201612.csv", encoding="cp1252", low_memory=False)

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
dftest['category'].value_counts()

# %%
dftest['main_category'].value_counts()

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

# %% [markdown]
# Campaign titles might include important information about the type of project, which could be predictive of success. To explore this, we want to extract the most common meaningful words from the campaign titles and create new features.  

# %%
df['lower_title'] = df['name'].str.lower()
df['words'] = df['lower_title'].str.split()
#turn all words into one big series
all_words = df['words'].explode()
word_counts = all_words.value_counts()
#remove meaningless words
stopwords = ['the', 'a', 'of', 'and', 'for', 'to', 'in', '&', '-', '(canceled)', 'by', 'your', 'with', 'on', 'an', 'my', 'new', 'from', 'first', 'short', 'is', 'you', 'help', 'at']
filtered_words = word_counts[~word_counts.index.isin(stopwords)]
print(filtered_words.head(10))

# %%
top_words = ['album', 'film', 'project', 'book', 'game', 'art', 'music', 'debut', 'documentary', 'life']
for word in top_words:
    df[f'has_{word}'] = df['lower_title'].str.contains(rf'\b{word}\b', na=False).astype(int)

#show the new columns
df.columns

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
X = df[['goal', 'backers', 'duration_days']]
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

# %% [markdown]
# Exclude backers and include title name features 

# %%
#define predictor variables and target
X = df[['goal', 'duration_days'] + [f'has_{w}' for w in top_words]]
y = df['state_encoded']

#Test Train Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#baseline accuracy
baseline_accuracy = y_train.value_counts(normalize=True).max()
print("Baseline accuracy:", baseline_accuracy)

#scale predictor variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#KNN
for k in [3, 5, 7, 9, 11 , 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    print(f"K={k} - Cross-val accuracy: {scores.mean():.3f}")

#logistic regression
logreg = LogisticRegression(max_iter=1000)

#cross validation
cv_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5)
print("Cross-val accuracy:", cv_scores.mean())

# %% [markdown]
# We apply forward feature selection to identify top 5 features 

# %%
#define all features (goal, duration_days, title keywords)
X = df[['goal', 'duration_days'] + [f'has_{w}' for w in top_words]]
y = df['state_encoded']

#Test Train Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#scale predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#logistic regression
logreg = LogisticRegression(max_iter=1000)

#forward feature selection
selector = SequentialFeatureSelector(logreg, n_features_to_select=5, direction='forward', cv=5)
selector.fit(X_train_scaled, y_train)

#print selected features
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]
print("Selected features:", list(selected_features))

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
