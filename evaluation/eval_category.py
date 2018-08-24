import sys
import yaml
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np

labels_true = []
labels_word = []
labels_pred = []

cat0 = yaml.load(open('mcrae_typicality.yaml'))

emb = {}
for line in open(sys.argv[1]):
    line = line.strip().split()
    emb[line[0]] = [float(i) for i in line[1:]]

cat = {}
vv = {}
num = 0
for i in cat0:
    if not i in cat:
	cat[i] = []
	vv[i] = num
	num += 1
    for j in cat0[i]:
	if j in emb.keys():
	    cat[i].append(j)
	    labels_true.append(vv[i])
	    labels_word.append(j)

X = []
for w in labels_word:
    X.append(emb[w])

X=np.array(X)

kmeans = KMeans(n_clusters=41, random_state=0).fit(X)
#print kmeans.labels_
labels_pred = list(kmeans.labels_)

print metrics.adjusted_rand_score(labels_true, labels_pred),
print metrics.adjusted_mutual_info_score(labels_true, labels_pred),
print metrics.normalized_mutual_info_score(labels_true, labels_pred)
