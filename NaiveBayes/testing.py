import numpy as np
from sklearn.naive_bayes import GaussianNB
import prep_terrain_data as prep
features_train, labels_train, features_test, labels_test = prep.makeTerrainData()
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
clf_pf.partial_fit(X, Y, np.unique(Y))

print(clf.fit(X,Y))