import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tabulate import tabulate

weekly = sm.datasets.get_rdataset("Weekly", package="ISLR")
print(weekly.data.info())

print(weekly.data.describe())

weekly.data["Direction"].unique()

spm = sns.pairplot(weekly.data)
spm.fig.set_size_inches(12, 12)
spm.savefig("img/4.10.a_pair.png")

corr_mat = weekly.data[weekly.data.columns[:-1]].corr()

fig, ax = plt.subplots(figsize=(8, 6))
cmap = sns.diverging_palette(220, 10, sep=80, n=7)
mask = np.triu(np.ones_like(corr_mat, dtype=np.bool))

with sns.axes_style("white"):
    sns.heatmap(corr_mat, cmap=cmap, mask=mask, robust=True, annot=True, ax=ax)

plt.tight_layout()
fig.savefig("img/4.10.a_corr_mat.png", dpi=90)
plt.close()

weekly_data = weekly.data.copy()
weekly_data["Direction"] = weekly.data["Direction"].map({"Up": 1, "Down": 0})

logit_model = smf.logit("Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume",
                        data=weekly_data).fit()
print(logit_model.summary())

def make_confusion_matrix_heatmap(conf_mat, categories, ax):
    """
    Makes a heat map visualization of the confusion matrix.
    """
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = [f"{value:.0f}" for value in conf_mat.flatten()]
    group_percentages = [f"{value:.2%}" for value in
                         conf_mat.flatten()/np.sum(conf_mat)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    with sns.axes_style("white"):
        sns.heatmap(conf_mat, cmap="Blues", fmt="", annot=labels, cbar=False,
                    xticklabels=categories, yticklabels=categories, ax=ax)

conf_mat = logit_model.pred_table(0.5)

fig, ax = plt.subplots(figsize=(4, 4))
categories = ["Down", "Up"]
make_confusion_matrix_heatmap(conf_mat, categories=categories, ax=ax)
plt.tight_layout()
fig.savefig("img/4.10.c_conf_mat.png", dpi=90)
plt.close()

accuracy = (np.sum(np.diag(conf_mat))) / np.sum(conf_mat)
print(f"Accuracy: {accuracy:.2%}")

weekly_training_set = weekly_data[weekly_data["Year"].between(1990, 2008)]
weekly_test_set = weekly_data.drop(weekly_training_set.index)

logit_model2 = smf.logit("Direction ~ Lag2", data=weekly_training_set).fit()
print(logit_model2.summary())

pred = np.array(logit_model2.predict(weekly_test_set["Lag2"]) > 0.5, dtype="int")
conf_mat2 = confusion_matrix(weekly_test_set["Direction"], pred)

fig, ax = plt.subplots(figsize=(4, 4))
categories = ["Down", "Up"]
make_confusion_matrix_heatmap(conf_mat2, categories=categories, ax=ax)
plt.tight_layout()
fig.savefig("img/4.10.d_conf_mat.png", dpi=90)
plt.close()

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(weekly_training_set["Lag2"].values[:, None],
              weekly_training_set["Direction"].values)

preds = lda_model.predict(weekly_test_set["Lag2"].values[:, None])
conf_mat_lda = confusion_matrix(weekly_test_set["Direction"], preds)

fig, ax = plt.subplots(figsize=(4, 4))
categories =["Down", "Up"]
make_confusion_matrix_heatmap(conf_mat_lda, categories, ax)
plt.tight_layout()
fig.savefig("img/4.10.e_conf_mat.png", dpi=90)
plt.close()

qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(weekly_training_set["Lag2"].values[:, None],
              weekly_training_set["Direction"].values)

preds = qda_model.predict(weekly_test_set["Lag2"].values[:, None])
conf_mat_qda = confusion_matrix(weekly_test_set["Direction"], preds)

fig, ax = plt.subplots(figsize=(4, 4))
categories =["Down", "Up"]
make_confusion_matrix_heatmap(conf_mat_qda, categories, ax)
plt.tight_layout()
fig.savefig("img/4.10.f_conf_mat.png", dpi=90)
plt.close()

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(weekly_training_set["Lag2"].values[:, None],
              weekly_training_set["Direction"].values)

preds = knn_model.predict(weekly_test_set["Lag2"].values[:, None])
conf_mat_knn = confusion_matrix(weekly_test_set["Direction"], preds)

fig, ax = plt.subplots(figsize=(4, 4))
categories =["Down", "Up"]
make_confusion_matrix_heatmap(conf_mat_knn, categories, ax)
plt.tight_layout()
fig.savefig("img/4.10.g_conf_mat.png", dpi=90)
plt.close()

auto_data = sm.datasets.get_rdataset("Auto", "ISLR").data

median_mileage = auto_data["mpg"].median()
auto_data["mpg01"] = np.where(auto_data["mpg"] > median_mileage, 1, 0)
print(tabulate(auto_data.head(), auto_data.columns, tablefmt="orgtbl"))

corr_mat = auto_data[auto_data.columns[1:]].corr()

cmap = "RdBu"
mask = np.triu(np.ones_like(corr_mat, dtype=np.bool))

fig, ax = plt.subplots(figsize=(8, 6))
with sns.axes_style("white"):
    sns.heatmap(corr_mat, cmap=cmap, mask=mask, annot=True, robust=True, ax=ax)

plt.tight_layout()
fig.savefig("img/4.11.b_corr_mat.png", dpi=90)
plt.close()

fig, axs = plt.subplots(4, 2, figsize=(12, 10))
sns.boxplot(y="cylinders", x="mpg01", data=auto_data, ax=axs[0, 0])
sns.boxplot(y="displacement", x="mpg01", data=auto_data, ax=axs[0, 1])
sns.boxplot(y="horsepower", x="mpg01", data=auto_data, ax=axs[1, 0])
sns.boxplot(y="weight", x="mpg01", data=auto_data, ax=axs[1, 1])
sns.boxplot(y="acceleration", x="mpg01", data=auto_data, ax=axs[2, 0])
sns.boxplot(y="year", x="mpg01", data=auto_data, ax=axs[2, 1])
sns.boxplot(y="origin", x="mpg01", data=auto_data, ax=axs[3, 0])
plt.tight_layout()
fig.savefig("img/4.11.b_box_plots.png")
plt.close()

X = auto_data[["cylinders", "displacement", "horsepower", "weight"]]
y = auto_data["mpg01"]
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(train_X, train_y)

test_pred = lda_model.predict(test_X)
conf_mat = confusion_matrix(test_pred, test_y)

fig, ax = plt.subplots(figsize=(4, 4))
make_confusion_matrix_heatmap(conf_mat, [0, 1], ax)
fig.savefig("img/4.11.d_conf_mat.png", dpi=90)
plt.close()

train_X_cyl = train_X["cylinders"]
test_X_cyl = test_X["cylinders"]

lda_model_cyl = LinearDiscriminantAnalysis()
lda_model_cyl.fit(train_X_cyl.values[:, None], train_y)

test_pred_cyl = lda_model_cyl.predict(test_X_cyl.values[:, None])
conf_mat_cyl = confusion_matrix(test_pred_cyl, test_y)

fig, ax = plt.subplots(figsize=(4, 4))
make_confusion_matrix_heatmap(conf_mat_cyl, [0, 1], ax)
fig.savefig("img/4.11.d_conf_mat_cyl.png", dpi=90)
plt.close()

qda_model_cyl = QuadraticDiscriminantAnalysis()
qda_model_cyl.fit(train_X_cyl.values[:, None], train_y)

test_pred_cyl = qda_model_cyl.predict(test_X_cyl.values[:, None])
conf_mat_cyl_qda = confusion_matrix(test_pred_cyl, test_y)

fig, ax = plt.subplots(figsize=(4, 4))
make_confusion_matrix_heatmap(conf_mat_cyl_qda, [0, 1], ax)
fig.savefig("img/4.11.d_conf_mat_cyl_qda.png", dpi=90)
plt.close()

qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(train_X, train_y)

test_pred = qda_model.predict(test_X)
conf_mat_qda = confusion_matrix(test_pred, test_y)

fig, ax = plt.subplots(figsize=(4, 4))
make_confusion_matrix_heatmap(conf_mat_qda, [0, 1], ax)
fig.savefig("img/4.11.d_conf_mat_qda.png", dpi=90)
plt.close()

logit_model = LogisticRegression()
logit_model.fit(train_X_cyl.values[:, None], train_y)

test_pred = logit_model.predict(test_X_cyl.values[:, None])
conf_mat_logit = confusion_matrix(test_pred, test_y)

fig, ax = plt.subplots(figsize=(4, 4))
make_confusion_matrix_heatmap(conf_mat_logit, [0, 1], ax)
fig.savefig("img/4.11.d_conf_mat_logit.png", dpi=90)
plt.close()

for k in np.logspace(0, 2, num=3, dtype=int):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_X_cyl.values[:, None], train_y)
    acc = accuracy_score(knn_model.predict(test_X_cyl.values[:, None]), test_y)
    print(f"K: {k:3}, Accuracy: {acc:.2%}, Test error: {1 - acc:.2f}")

for k in np.logspace(0, 2, num=3, dtype=int):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_X, train_y)
    acc = accuracy_score(knn_model.predict(test_X), test_y)
    print(f"K: {k:3}, Accuracy: {acc:.2%}, Test error: {1 - acc:.2f}")

boston_data = sm.datasets.get_rdataset("Boston", "MASS").data
print(tabulate(boston_data.head(), boston_data.columns, tablefmt="orgtbl"))

median_crim = boston_data["crim"].median()
boston_data["crim01"] = np.where(boston_data["crim"] > median_crim, 1, 0)

corr_mat = boston_data[boston_data.columns[1:]].corr()
fig, ax = plt.subplots(figsize=(10, 8))
cmap = "RdBu"
mask = np.triu(np.ones_like(corr_mat, dtype=np.bool))
sns.heatmap(corr_mat, cmap=cmap, mask=mask, annot=True, robust=True, ax=ax)
plt.tight_layout()
fig.savefig("img/4.13_corr_mat.png", dpi=90)
plt.close()

X = boston_data[["indus", "nox", "age", "rad", "tax", "dis"]]
y = boston_data["crim01"]

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

logit_model = LogisticRegression()
logit_model.fit(train_X, train_y)

acc = accuracy_score(logit_model.predict(test_X), test_y)
print(f"Accuracy: {acc:.2f}, Test error: {1 - acc:.2f}")

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(train_X, train_y)

acc = accuracy_score(lda_model.predict(test_X), test_y)
print(f"Accuracy: {acc:.2f}, Test error: {1 - acc:.2f}")

for k in np.logspace(0, 2, num=3, dtype=int):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_X, train_y)
    acc = accuracy_score(knn_model.predict(test_X), test_y)
    print(f"K: {k:3}, Accuracy: {acc:.2f}, Test error: {1 - acc:.2f}")

train_X_reduced = train_X[["indus", "dis", "tax"]]
test_X_reduced = test_X[["indus", "dis", "tax"]]

logit_model = LogisticRegression()
logit_model.fit(train_X_reduced, train_y)

acc = accuracy_score(logit_model.predict(test_X_reduced), test_y)
print(f"Logistic regression: Accuracy: {acc:.2f}, Test error: {1 - acc:.2f}")

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(train_X_reduced, train_y)

acc = accuracy_score(lda_model.predict(test_X_reduced), test_y)
print(f"LDA: Accuracy: {acc:.2f}, Test error: {1 - acc:.2f}")

print("KNN:")
for k in np.logspace(0, 2, num=3, dtype=int):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_X_reduced, train_y)
    acc = accuracy_score(knn_model.predict(test_X_reduced), test_y)
    print(f"K: {k:3}, Accuracy: {acc:.2f}, Test error: {1 - acc:.2f}")
