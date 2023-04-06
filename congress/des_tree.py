import pandas as p
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt

df = p.read_csv("CongressionalVotingID.shuf.lrn.csv")
df_result = p.read_csv("CongressionalVotingID.shuf.tes.csv")
expected = p.read_csv("CongressionalVotingID.shuf.sol.ex.csv")


df_removed_column = df.drop(columns=["class", "ID"])
X = df_removed_column.replace({"unknown": np.nan})

ordinal_encoder = OrdinalEncoder()
X = ordinal_encoder.fit_transform(X)

imputer = KNNImputer(n_neighbors=5, weights="uniform")
X = imputer.fit_transform(X)

y = df["class"]

df_result = df_result.drop(columns=["ID"])

df_result = df_result.replace({"unknown": np.nan})
df_result = ordinal_encoder.transform(df_result)
df_result = imputer.transform(df_result)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

plt.figure()
tree.plot_tree(clf, filled=True)
plt.title("decision tree")
plt.savefig('tree.png', dpi=400)

predicted = clf.predict(df_result)
expected["class"] = predicted
expected.to_csv("result_desission_tree.csv", index=False)
# acc = accuracy_score(predicted, expected["class"])
# print(acc)
