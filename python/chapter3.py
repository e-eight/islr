import pandas as pd
from tabulate import tabulate

df = pd.DataFrame.from_dict({'X1': [0, 2, 0, 0, -1, 1], 'X2': [3, 0, 1, 1, 0, 1], 'X3': [0, 0, 3, 2, 1, 1], 'Y':['Red', 'Red', 'Red', 'Green', 'Green', 'Red']})
test = np.array([0, 0, 0])
df['Distance'] = np.linalg.norm(df[['X1', 'X2', 'X3']].values-test, axis=1)
pd.set_option('precision', 5)
print(tabulate(df, df.columns, tablefmt="orgtbl"))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

flexibility = np.linspace(0, 10, 100)
squared_bias = 0.02 * (10 - flexibility) ** 2
variance = 0.02 * flexibility ** 2
training_error = 0.003 * (10 - flexibility) ** 3
test_error = 3 - 0.6 * flexibility + 0.06 * flexibility ** 2
bayes_error = np.ones_like(flexibility)

plt.close('all') # To prevent memory consumption
fig, ax = plt.subplots()
ax.plot(flexibility, squared_bias, label="Bias")
ax.plot(flexibility, variance, label="Variance")
ax.plot(flexibility, training_error, label="Training Error")
ax.plot(flexibility, test_error, label="Test Error")
ax.plot(flexibility, bayes_error, label="Bayes Error")
ax.set_xlabel("Flexibility")
ax.legend(loc="upper center")

sns.despine()

fig.savefig("img/bv-decomp.png", dpi=90)

college = pd.read_csv("data/College.csv")
print(tabulate(college.head(), college.columns, tablefmt="orgtbl"))

college.set_index("Unnamed: 0", inplace=True)
college.index.name = "Names"

headers = [college.index.name] + list(college.columns)
print(tabulate(college.head(), headers, tablefmt="orgtbl"))

print(tabulate(college.describe(), college.columns, tablefmt="orgtbl"))

plot_columns = list(college.columns)[:10]
plt.close('all')
spm = sns.pairplot(college[plot_columns])
spm.fig.set_size_inches(12, 12)
spm.savefig("img/college_scatter.png", dpi=90)

plt.close('all')
bp1 = sns.boxplot(x="Private", y="Outstate", data=college)
sns.despine()
plt.tight_layout()
bp1.get_figure().savefig("img/college_outstate_private.png", dpi=90)

college["Elite"] = college["Top10perc"].apply(lambda x: "Yes" if x > 50 else "No")
print(college["Elite"].value_counts())

plt.close('all')
bp2 = sns.boxplot(x="Elite", y="Outstate", data=college)
sns.despine()
plt.tight_layout()
bp2.get_figure().savefig("img/college_outstate_elite.png", dpi=90)

print(college.info())

cut_bins3 = ["Low", "Medium", "High"]
cut_bins5 = ["Very Low", "Low", "Medium", "High", "Very High"]
college["Enroll2"] = pd.cut(college["Enroll"], 5, labels=cut_bins5)
college["Books2"] = pd.cut(college["Books"], 3, labels=cut_bins3)
college["PhD2"] = pd.cut(college["PhD"], 3, labels=cut_bins3)
college["Grad.Rate2"] = pd.cut(college["Grad.Rate"], 5, labels=cut_bins5)

plt.close("all")
fig, axs = plt.subplots(2, 2)
sns.countplot(college["Enroll2"], ax=axs[0, 0])
sns.countplot(college["Books2"], ax=axs[0, 1])
sns.countplot(college["PhD2"], ax=axs[1, 0])
sns.countplot(college["Grad.Rate2"], ax=axs[1, 1])
sns.despine()

axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=40, ha="right")
axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=40, ha="right")
axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=40, ha="right")
axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=40, ha="right")

plt.subplots_adjust(wspace=0.4, hspace=1)
fig.savefig("img/college_hist.png", dpi=90)

auto = pd.read_csv("data/Auto.csv")
auto.dropna(inplace=True)
print(auto.info())

print(auto["horsepower"].unique())

auto.drop(auto[auto.horsepower == "?"].index, inplace=True)
auto["horsepower"] = pd.to_numeric(auto["horsepower"])
print(auto.info())

from pprint import pprint

quant = auto.select_dtypes(exclude="object").columns
ranges = {col: (min(auto[col]), max(auto[col])) for col in quant}
pprint(ranges)

msd = {col: {"mean": round(np.mean(auto[col]), 2), "std": round(np.std(auto[col]), 2)} for col in quant}
pprint(msd)

# An alternative is to use the following aggregrate method:
# auto.agg(["mean", "std"])

auto2 = auto.drop(auto.index[10:85])

ranges = {col: (min(auto2[col]), max(auto2[col])) for col in quant}
pprint(ranges)

msd = {col: {"mean": round(np.mean(auto[col]), 2), "std": round(np.std(auto[col]), 2)} for col in quant}
pprint(msd)

plt.close('all')
spm = sns.pairplot(auto[["mpg", "horsepower", "weight", "displacement", "acceleration"]])
spm.fig.set_size_inches(6, 6)
spm.savefig("img/auto_pair.png")

from sklearn.datasets import load_boston

lb = load_boston()
boston = pd.DataFrame(lb.data, columns=lb.feature_names)
boston['MEDV'] = lb.target
print(tabulate(boston.head(), boston.columns, tablefmt="orgtbl"))

print(lb['DESCR'])

plt.close("all")
spm = sns.pairplot(boston, plot_kws = {'s': 10})
spm.fig.set_size_inches(12, 12)
spm.savefig("img/boston_scatter.png", dpi=90)

print(boston.corrwith(boston["CRIM"]).sort_values())

plt.close("all")
sns.scatterplot(x="TAX", y="CRIM", data=boston)
sns.despine()
plt.tight_layout()
plt.savefig("img/boston_crim_tax.png", dpi=90)

plt.close("all")
sns.boxplot(x="RAD", y="CRIM", data=boston)
sns.despine()
plt.tight_layout()
plt.savefig("img/boston_crim_rad.png", dpi=90)

ranges = {col: (boston[col].min(), boston[col].max()) for col in boston.columns[:-1]}
pprint(ranges)

high_crime = boston.nlargest(5, "CRIM")
print(tabulate(high_crime, boston.columns, tablefmt="orgtbl"))

high_tax = boston.nlargest(5, "TAX")
print(tabulate(high_tax, boston.columns, tablefmt="orgtbl"))

print(boston["CHAS"].value_counts())

print(boston["PTRATIO"].median())

print(tabulate(boston.nsmallest(1, "MEDV"), boston.columns, tablefmt="orgtbl"))

print(tabulate(boston.describe(), boston.columns, tablefmt="orgtbl"))

rm7 = np.sum(boston["RM"] > 7)
rm8 = np.sum(boston["RM"] > 8)
print(rm7, rm8)

eight_rooms = boston[boston["RM"] > 8]
print(tabulate(eight_rooms.describe(), boston.columns, tablefmt="orgtbl"))
