import pandas as p
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

df = p.read_csv("CongressionalVotingID.shuf.lrn.csv")
df_result = p.read_csv("CongressionalVotingID.shuf.tes.csv")
expected = p.read_csv("CongressionalVotingID.shuf.sol.ex.csv")


df_removed_column = df.drop(columns=["class", "ID"])
X = df_removed_column.replace({"y": 1, "n": 0, "unknown": None})
imputer = KNNImputer(n_neighbors=8, weights="uniform")
X = imputer.fit_transform(X)

y = df["class"]

df_result = df_result.drop(columns=["ID"])
df_result = df_result.fillna("unknown")
df_result = df_result.replace({"y": 1, "n": 0, "unknown": None})
df_result = imputer.transform(df_result)

clf = GaussianNB()
clf = clf.fit(X, y)

redicted = clf.predict(df_result)
acc = accuracy_score(predicted, expected["class"])
print(acc)
