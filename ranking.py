import math
import numpy
from operator import itemgetter
from numpy.linalg import norm
import yaml
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np

EPSILON = 1e-6

def euclidean(vec1, vec2):
  diff = vec1 - vec2
  return math.sqrt(diff.dot(diff))

def cosine_sim(vec1, vec2):
  vec1 += EPSILON * numpy.ones(len(vec1))
  vec2 += EPSILON * numpy.ones(len(vec1))
  return vec1.dot(vec2)/(norm(vec1)*norm(vec2))

def assign_ranks(item_dict):
  ranked_dict = {}
  sorted_list = [(key, val) for (key, val) in sorted(item_dict.items(),
                                                     key=itemgetter(1),
                                                     reverse=True)]
  for i, (key, val) in enumerate(sorted_list):
    same_val_indices = []
    for j, (key2, val2) in enumerate(sorted_list):
      if val2 == val:
        same_val_indices.append(j+1)
    if len(same_val_indices) == 1:
      ranked_dict[key] = i+1
    else:
      ranked_dict[key] = 1.*sum(same_val_indices)/len(same_val_indices)
  return ranked_dict

def correlation(dict1, dict2):
  avg1 = 1.*sum([val for key, val in dict1.iteritems()])/len(dict1)
  avg2 = 1.*sum([val for key, val in dict2.iteritems()])/len(dict2)
  numr, den1, den2 = (0., 0., 0.)
  for val1, val2 in zip(dict1.itervalues(), dict2.itervalues()):
    numr += (val1 - avg1) * (val2 - avg2)
    den1 += (val1 - avg1) ** 2
    den2 += (val2 - avg2) ** 2
  return numr / math.sqrt(den1 * den2)

def spearmans_rho(ranked_dict1, ranked_dict2):
  assert len(ranked_dict1) == len(ranked_dict2)
  if len(ranked_dict1) == 0 or len(ranked_dict2) == 0:
    return 0.
  x_avg = 1.*sum([val for val in ranked_dict1.values()])/len(ranked_dict1)
  y_avg = 1.*sum([val for val in ranked_dict2.values()])/len(ranked_dict2)
  num, d_x, d_y = (0., 0., 0.)
  for key in ranked_dict1.keys():
    xi = ranked_dict1[key]
    yi = ranked_dict2[key]
    num += (xi-x_avg)*(yi-y_avg)
    d_x += (xi-x_avg)**2
    d_y += (yi-y_avg)**2
  return num/(math.sqrt(d_x*d_y))


def eval_category(word_vecs):
    labels_true = []
    labels_word = []

    cat0 = yaml.load(open('evaluation/mcrae_typicality.yaml'))

    emb = {}
    nn = 0
    for line in word_vecs:
        emb[nn] = line
        nn += 1

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

    X = np.array(X)

    kmeans = KMeans(n_clusters=41, random_state=0).fit(X)
    # print kmeans.labels_
    labels_pred = list(kmeans.labels_)

    r1 = metrics.adjusted_rand_score(labels_true, labels_pred),
    r2 = metrics.adjusted_mutual_info_score(labels_true, labels_pred),
    r3 = metrics.normalized_mutual_info_score(labels_true, labels_pred)

    return r1, r2, r3