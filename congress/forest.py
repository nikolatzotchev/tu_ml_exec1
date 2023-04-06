import pandas as p
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn import tree
import matplotlib.pyplot as plt

df = p.read_csv("CongressionalVotingID.shuf.lrn.csv")
df_result = p.read_csv("CongressionalVotingID.shuf.tes.csv")
expected = p.read_csv("CongressionalVotingID.shuf.sol.ex.csv")

df_removed_column = df.drop(columns=["class", "ID"])
X = df_removed_column.replace({"y": 1, "n": 0, "unknown": None})
imputer = KNNImputer(n_neighbors=4, weights="uniform")
X = imputer.fit_transform(X)

y = df["class"]

df_result = df_result.drop(columns=["ID"])
df_result = df_result.fillna("unknown")
df_result = df_result.replace({"y": 1, "n": 0, "unknown": None})
df_result = imputer.fit_transform(df_result)

clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
clf = clf.fit(X, y)

predicted = clf.predict(df_result)
acc = accuracy_score(predicted, expected["class"])
print(acc)

plt.figure()
estimator = clf.estimators_[0]
tree.plot_tree(estimator, filled=True)
plt.title("forest example")
plt.savefig('forest.png', dpi=400)
