import pandas as pd
from sklearn import preprocessing

# Read the data
pubs = pd.read_csv('armenian_pubs.csv')
print(pubs.info())
print(pubs.isnull().sum())


# select coloumns for analis
#col_rank = ['Occupation', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color']
#mush = mush[col_rank]
col_rank = pubs.columns

# encoding data
ordinal_encoder = preprocessing.OrdinalEncoder(dtype=int)
pubs = pd.DataFrame(ordinal_encoder.fit_transform(pubs), columns=col_rank)

print('\n', pubs)

data = pubs.drop('Occupation', axis=1)

#--------------------------------------------------
# K-mode algoritm

POINT_N = len(data)
DIM_N = 16
CLUST_N = 2

class CLUST:
    def __init__(self, x, c):
        self.X = x
        self.clust = c
    def dist(self, X):
        d = (self.X != X).sum()
        return d
    def eval(self, df):
        cols = df.columns
        self.X = []
        for c in cols:
            self.X.append( df[c].value_counts().idxmax() )
#        c = cols[0]
#        print( df[c].value_counts() )
        return

Clust = [CLUST(data.values[0], pubs['Occupation'][1])]
i0 = 0
d0 = 0
for i in range(0, POINT_N):
    d = Clust[0].dist(data.values[i])
    if d > d0 and Clust[0].clust != pubs['Occupation'][i]:
        d0 = d
        i0 = i
Clust.append(CLUST(data.values[i0], pubs['Occupation'][i0]))
i0 = 0
d0 = 0

print(Clust[0].X)
print(Clust[0].clust)
print(Clust[1].X)
print(Clust[1].clust)

Res = pd.DataFrame(data=[CLUST_N for i in range(0, POINT_N)], columns=['clust'])

for n in range(0,3):
    for i in range(0, POINT_N):
        if Clust[0].dist(data.values[i]) < Clust[1].dist(data.values[i]):
            Res['clust'][i] = Clust[0].clust
        else:
            Res['clust'][i] = Clust[1].clust
    for cl in Clust:
        df = data.loc[Res['clust'] == cl.clust]
        cl.eval(df)
        print('\n', cl.X)

    r = Res['clust'] != pubs['Occupation']
    print('\n', r.sum()/len(r))

#--------------------------------------------------
# Importing Libraries
from kmodes.kmodes import KModes
print('\n')
print('\n')

#Using K-Mode with "Cao" initialization
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_data = km_cao.fit_predict(data)
print('\n')
print(fitClusters_data)

clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = data.columns
print(clusterCentroidsDf)

res = fitClusters_data == pubs['Occupation']
print('\n', res.sum()/len(res))