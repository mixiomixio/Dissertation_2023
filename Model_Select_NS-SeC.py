import pandas as pd
from sklearn.model_selection import train_test_split
#from pysal.model import spreg
##*********************************************************************************************

data = pd.read_csv("!NS-SeC.csv")
data = data.dropna()

X = data.iloc[:,6:16]
y = data.iloc[:,2]

X['L1, L2 and L3'] = X['L1, L2 and L3']/X['Sum']*100
X['L4, L5 and L6'] = X['L4, L5 and L6']/X['Sum']*100
X['L7'] = X['L7']/X['Sum']*100
X['L8 and L9'] = X['L8 and L9']/X['Sum']*100
X['L10 and L11'] = X['L10 and L11']/X['Sum']*100
X['L12'] = X['L12']/X['Sum']*100
X['L13'] = X['L13']/X['Sum']*100
X['L14.1 and L14.2'] = X['L14.1 and L14.2']/X['Sum']*100
X['L15'] = X['L15']/X['Sum']*100

X = X[['L7', 'L10 and L11', 'L12', 'L14.1 and L14.2', 'L15']]
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5)

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
print(linear_regression.coef_)
y_pred = linear_regression.predict(X_test)
print("LR", linear_regression.score(X_test, y_test))

from sklearn.linear_model import Lasso
reg = Lasso()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("Lasso", reg.score(X_test, y_test))

from sklearn import svm
regr = svm.SVR()
regr.fit(X_train, y_train)
regr.predict(X_test)
print("SVM", regr.score(X_test, y_test))

from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
clf.predict(X_test)
print("TREE", clf.score(X_test, y_test))

from sklearn.neural_network import MLPRegressor
regrr = MLPRegressor()
regrr.fit(X_train, y_train)
#print("MLP COE", regrr.coefs_)
regrr.predict(X_test)
print("MLP", regrr.score(X_test, y_test))

from sklearn.kernel_ridge import KernelRidge
krr = KernelRidge()
krr.fit(X_train, y_train)
krr.predict(X_test)
print("KernelRidge", krr.score(X_test, y_test))

from sklearn.cross_decomposition import PLSRegression
pls2 = PLSRegression()
pls2.fit(X_train, y_train)
pls2.predict(X_test)
print("PLS", pls2.score(X_test, y_test))

from sklearn.linear_model import ElasticNet
rex = ElasticNet(random_state=1)
rex.fit(X_train, y_train)
rex.predict(X_test)
print("ElaNet", rex.score(X_test, y_test))

from sklearn import linear_model
clf1 = linear_model.PoissonRegressor()
clf1.fit(X_train, y_train)
clf1.predict(X_test)
print("Poi", clf1.score(X_test, y_test))

'''
CVscore = cross_validate(regr, X, y, groups=X.index.values, cv=10, n_jobs=-1)
print(np.mean(CVscore['test_score']))
'''