# Insurance data HDBSCAN
Finance = pd.read_csv("D:/Hierarchial clustering/Insurance Cluster/Finance Cluster.csv", index_col='keyword')  

# Needed imports
import hdbscan

# Other needed imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import collections
%matplotlib inline
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}

# Visualting data
from sklearn.manifold import TSNE
projection = TSNE().fit_transform(Finance)
plt.scatter(*projection.T, **plot_kwds)


# cluster_selection_method 
cluster = hdbscan.HDBSCAN(min_cluster_size=240, min_samples=2,cluster_selection_method='leaf')
cluster_labels = cluster.fit_predict(Finance)

# cl = cluster.fit(sample)
cluster.labels_

cluster.labels_.max()
#cluster.probabilities_

df = pd.DataFrame(cluster_labels)

import collections
counter=collections.Counter(cluster.labels_)
print(counter)

test = pd.concat([Insurance,df], axis=1)

test.to_csv('D:/Hierarchial clustering/Insurance Cluster/out_120_1_L.csv')
