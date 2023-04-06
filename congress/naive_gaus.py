import pandas as p
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt

df = p.read_csv("CongressionalVotingID.shuf.lrn.csv")
df_result = p.read_csv("CongressionalVotingID.shuf.tes.csv")
expected = p.read_csv("CongressionalVotingID.shuf.sol.ex.csv")


y = df["class"]

df_removed_column = df.drop(columns=["class", "ID"])
X = df_removed_column.replace({"unknown": np.nan})

ordinal_encoder = OrdinalEncoder()
X = ordinal_encoder.fit_transform(X)

imputer = KNNImputer(n_neighbors=4, weights="uniform")
X = imputer.fit_transform(X)
print(X)


df_result = df_result.drop(columns=["ID"])
# df_result = df_result.fillna("unknown")
df_result = df_result.replace({"unknown": np.nan})
df_result = ordinal_encoder.transform(df_result)
df_result = imputer.transform(df_result)


clf = GaussianNB()
clf = clf.fit(X, y)

predicted = clf.predict(df_result)
print(predicted)
expected["class"] = predicted
print(expected)
expected.to_csv("result_gauss.csv", index=False)
