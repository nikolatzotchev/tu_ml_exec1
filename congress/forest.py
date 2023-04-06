import pandas as p
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree

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

df_result = df_result.drop(columns=["ID"])
df_result = df_result.replace({"unknown": np.nan})
df_result = ordinal_encoder.transform(df_result);
df_result = imputer.fit_transform(df_result)

clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
clf = clf.fit(X, y)

predicted = clf.predict(df_result)
expected["class"] = predicted
expected.to_csv("forest_result.csv",index=False)

plt.figure()
estimator = clf.estimators_[0]
tree.plot_tree(estimator, filled=True)
plt.title("forest example")
plt.savefig('forest.png', dpi=400)
