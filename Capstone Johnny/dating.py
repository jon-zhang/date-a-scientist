import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

#Create your df here:
df = pd.read_csv("profiles.csv")

#assigning numbers to categories of body_type and income
body_mapping = {"rather not say": np.nan, "used up":0, "overweight":1, "full figured":2, "curvy":3, "a little extra":4, "average":5, "skinny":6, "thin": 7, "fit":8, "athletic":9, "jacked":10}
df["body_code"]=df.body_type.map(body_mapping)

income_mapping = {-1:np.nan, 20000:1, 30000:2, 40000:3, 50000:4,60000:5, 70000:6, 80000:7, 100000:8, 150000:9,250000:10, 5000000:11, 10000000:12}
df["income"]=df.income.map(income_mapping)

#testing out different columns
'''
print(df.job.head())
print(df.body_type.head())
print(df.status.head())
print(df.pets.head())
print(df.education.head())
print(df.drugs.head())
print(df.sex.head())
'''
#can your height and body_type predict your income?
#shows first graph
plt.hist(df.height.dropna(), bins=50)
plt.xlabel("Height (inches)")
plt.ylabel("Frequency")
plt.xlim(55,85)
plt.show()
plt.clear()


#show bar graph of body types by frequency
plt.clear()
df.body_type.value_counts().plot('bar')
plt.show()

#removnig all rows with NaNs and normalizing values
dfsub=df[['height', 'body_code','income']].dropna()
feature_data=dfsub[['height', 'body_code']]
feature_datatwo=dfsub[['height', 'body_code']]

x=feature_data.values
min_max_scaler=preprocessing.MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(x)
feature_data=pd.DataFrame(x_scaled, columns=feature_data.columns)


#fitting the K-Nearest neighbor Classifier
knscore=[]
knlist=list(range(1,101))
for k in range(100):
    classifier=KNeighborsClassifier(n_neighbors=k+1)
    classifier.fit(feature_data, dfsub.income)
    predictions=classifier.predict(feature_data)
    knscore.append(accuracy_score(dfsub.income, predictions))
print(precision_score(dfsub.income, predictions, average=None))
print(recall_score(dfsub.income, predictions, average=None))
plt.xlabel('k')
plt.ylabel('Accuracy Score')
plt.plot(knlist,knscore)
plt.show()

#fitting the SVC and scoring
svcscore=[]
svclist=list(range(1, 101))
for g in range(100):
    classifysvc=SVC(gamma=g+1)
    classifysvc.fit(feature_data, dfsub.income)
    svcscore.append(classifysvc.score(feature_data, dfsub.income))
plt.xlabel('gamma')
plt.ylabel('Score')
plt.plot(svclist,svcscore)
plt.show()

#multivariate linear regression
mlr=LinearRegression()
model=mlr.fit(feature_datatwo, dfsub.income)
y_predict=mlr.predict(feature_datatwo)
print(model.score(feature_datatwo, dfsub.income))
#plotting multivariate linear regression with y label vs y-predict
plt.scatter(dfsub.income, y_predict, alpha=0.1)
plt.xlabel('Actual Income')
plt.ylabel('Predicted Income')
plt.show()


#fitting and predicting the K-Nearest Neighbor regressor
regscore=[]
regklist=list(range(1,101))
for k in range(100):
    regressor=KNeighborsRegressor(n_neighbors = k+1)
    regressor.fit(feature_data, dfsub.income)
    regpredictions=regressor.predict(feature_data)
    regscore.append(r2_score(dfsub.income, regpredictions))

plt.xlabel('k')
plt.ylabel('R^2 Score')
plt.plot(regklist,regscore)
plt.show()