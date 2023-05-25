
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


df = pd.read_csv("heart_cleveland_upload.csv")

print(df.info())
'''
#df = pd.read_csv("seeds.csv", sep = ",")

Srv = df['condition'].values

colors = ['blue', 'lime', 'magenta', 'orange', 'yellow', 'cyan', 'darkviolet']

for i in range(0, len(df)): #нужно заменить в кволити на 0, 1 и 2
    plt.scatter(df['Height'][i], df['Diameter'][i], c = colors[Srv[i]], s=20)
plt.show()
'''

# min_margin_low = df["ShellWeight"].mean()
# df["ShellWeight"].fillna(min_margin_low, inplace=True)

km = KMeans(n_clusters=2)
km_predictions = km.fit_predict(df.drop(["condition"],axis = 1))

df_pred = df.assign(predicted=km_predictions)

#print(df_pred.info())

cluster_of_genuine = round(df_pred[df_pred["condition"] == True]["predicted"].mean())
predicted = 0
total = 0
for iter, t in df_pred.iterrows():
    total+=1
    if (t["predicted"] == cluster_of_genuine) == (t["condition"]):
        predicted+=1

print("Rate of prediction is ", predicted/total)

cntr, u, u0, d, jm, p, fpc = fuzz.cmeans((df.drop(["condition"],axis = 1).transpose()), c=2, m = 2, error=0.05, maxiter=10)

#Учитывая, что u[1] = 1 - u[0], можно хранить только u[0] или u[1]
df_fcm = df_pred.assign(cl0=u[0], cl1=u[1])

tr = df_fcm["cl0"].mean()
mean_genuine = df_fcm[df_fcm["condition"] == True]["cl0"].mean()
cl_genuine = 1
if (mean_genuine > tr):
    cl_genuine = 0


fpr, tpr, thresholds = metrics.roc_curve(y_true=df_fcm["condition"], y_score=
                                         (df_fcm["cl0"] if cl_genuine == 0 else df_fcm["cl1"]), 
                                         )

plt.plot (fpr,tpr)
#plt.show()




y = df.condition
X =  df.drop('condition', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)


targ_pred = knn.predict_proba(x_test)[:, 1]
knn.score(x_test,y_test)

fpr, tpr, thr = metrics.roc_curve (y_test, targ_pred)



plt.plot (fpr,tpr)
plt.show()







'''
Srv = df['is_genuine'].values

colors = ['blue', 'lime', 'magenta', 'orange', 'yellow', 'cyan', 'darkviolet']

for i in range(0, len(df)): #нужно заменить в кволити на 0, 1 и 2
    plt.scatter(df['length'][i], df['diagonal'][i], c = colors[Srv[i]], s=20)
plt.show()


min_margin_low = df["margin_low"].mean()
df["margin_low"].fillna(min_margin_low, inplace=True)

km = KMeans(n_clusters=2)
km_predictions = km.fit_predict(df.drop(["is_genuine"],axis = 1))

df_pred = df.assign(predicted=km_predictions)

#print(df_pred.info())

cluster_of_genuine = round(df_pred[df_pred["is_genuine"] == True]["predicted"].mean())
predicted = 0
total = 0
for iter, t in df_pred.iterrows():
    total+=1
    if (t["predicted"] == cluster_of_genuine) == (t["is_genuine"]):
        predicted+=1

print("Rate of prediction is ", predicted/total)

cntr, u, u0, d, jm, p, fpc = fuzz.cmeans((df.drop(["is_genuine"],axis = 1).transpose()), c=2, m = 2, error=0.05, maxiter=10)

#Учитывая, что u[1] = 1 - u[0], можно хранить только u[0] или u[1]
df_fcm = df_pred.assign(cl0=u[0], cl1=u[1])

tr = df_fcm["cl0"].mean()
mean_genuine = df_fcm[df_fcm["is_genuine"] == True]["cl0"].mean()
cl_genuine = 1
if (mean_genuine > tr):
    cl_genuine = 0
#print("-------")

#print(cl_genuine)

#print("-------")
#print(u)
#print("-------")
#print(u0)
#print("-------")
#print(d)
#print("-------")
#print(df_fcm)

fpr, tpr, thresholds = metrics.roc_curve(y_true=df_fcm["is_genuine"], y_score= 
                                         (df_fcm["cl0"] if cl_genuine == 0 else df_fcm["cl1"]), 
                                         )
plt.plot (fpr,tpr)
plt.plot (fpr,tpr)
#plt.show()




y = df.is_genuine
X =  df.drop('is_genuine', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)


targ_pred = knn.predict_proba(x_test)[:, 1]
scr = knn.score(x_test,y_test)

fpr, tpr, thr = metrics.roc_curve (y_test, targ_pred)

print("Rate of prediction is ", scr)


plt.plot (fpr,tpr)
plt.show()
#print(fpr)
#print(tpr)
#print(thresholds)
'''




'''
df = pd.read_csv("seeds.csv", sep = ",")

Srv = df['condition'].values

colors = ['blue', 'lime', 'magenta', 'orange', 'yellow', 'cyan', 'darkviolet']

for i in range(0, len(df)): #нужно заменить в кволити на 0, 1 и 2
    plt.scatter(df['Kernel.Length'][i], df['Kernel.Width'][i], c = colors[Srv[i]], s=20)
plt.show()


min_margin_low = df["Area"].mean()
df["Area"].fillna(min_margin_low, inplace=True)

km = KMeans(n_clusters=2)
km_predictions = km.fit_predict(df.drop(["condition"],axis = 1))

df_pred = df.assign(predicted=km_predictions)

#print(df_pred.info())

cluster_of_genuine = round(df_pred[df_pred["condition"] == True]["predicted"].mean())
predicted = 0
total = 0
for iter, t in df_pred.iterrows():
    total+=1
    if (t["predicted"] == cluster_of_genuine) == (t["condition"]):
        predicted+=1

print("Rate of prediction is ", predicted/total)

cntr, u, u0, d, jm, p, fpc = fuzz.cmeans((df.drop(["condition"],axis = 1).transpose()), c=2, m = 2, error=0.05, maxiter=10)

#Учитывая, что u[1] = 1 - u[0], можно хранить только u[0] или u[1]
df_fcm = df_pred.assign(cl0=u[0], cl1=u[1])

tr = df_fcm["cl0"].mean()
mean_genuine = df_fcm[df_fcm["condition"] == True]["cl0"].mean()
cl_genuine = 1
if (mean_genuine > tr):
    cl_genuine = 0
#print("-------")

#print(cl_genuine)

#print("-------")
#print(u)
#print("-------")
#print(u0)
#print("-------")
#print(d)
#print("-------")
#print(df_fcm)

fpr, tpr, thresholds = metrics.roc_curve(y_true=df_fcm["condition"], y_score= 
                                         (df_fcm["cl0"] if cl_genuine == 0 else df_fcm["cl1"]), 
                                         )
plt.plot (fpr,tpr)
#plt.show()




y = df.is_genuine
X =  df.drop('condition', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)


targ_pred = knn.predict_proba(x_test)[:, 1]
knn.score(x_test,y_test)

fpr, tpr, thr = metrics.roc_curve (y_test, targ_pred)



plt.plot (fpr,tpr)
plt.show()




'''