import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from Levenshtein import distance
import matplotlib.pyplot as plt
from collections import defaultdict

# specify the maximum Levenshtein distance between two sequences for them to be considered neighbors
eps = 1

# specify the minimum number of sequences in a cluster
min_samples = 2

# Read in the set of sequences
# Load CSV file into a Pandas DataFrame
df = pd.read_csv('Covid_19_sample.csv')

# Extract bio-sequences from DataFrame
sequences = df['rbd_dna'].tolist()
# ids = df['id'].tolist()
ids = [('/'.join(fields[1].split('/')[2:4]) + '/' + fields[1].split('/')[-1].split(',')[0]) 
       if len(fields[1].split('/')) > 1 else "Origin Not Defined" for fields in [i.split('|') for i in df['header'].tolist() if i != " "]]

#filling the ids with Origin Not defined if the data is redundant or empty
ids = ["Origin Not Defined" if ids[i] == " " or "genome" in ids[i] else ids[i] for i in range(len(ids))]

# Calculate the distances between the sequences using the Levenshtein distance metric
distances = np.zeros((len(sequences), len(sequences)))
for i in range(len(sequences)):
    for j in range(i+1, len(sequences)):
        dist = distance(sequences[i], sequences[j])
        distances[i,j] = dist
        distances[j,i] = dist

# perform DBSCAN clustering
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
dbscan.fit(distances)

# Calculate the Silhouette Score for the clustering
silhouette_avg = silhouette_score(distances, dbscan.labels_, metric='precomputed')
print("The average Silhouette Score is :", silhouette_avg)

#create a dictionary to store the cluster IDs and sequence IDs
clusters = defaultdict(list)
for i, label in enumerate(dbscan.labels_):
    clusters[label].append(ids[i])

#Declaring the variable total to verify the count of data points
total=0

# print the clusters
m = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0) # number of clusters
print(f"Number of clusters: {m}")
for i, sequence_ids in clusters.items():
    if i == -1:
        continue
    print(f"Cluster {i}: {len(sequence_ids)} sequences")
    total+= len(sequence_ids)
    print(clusters[i]) 
print(f"Outliers :{len(clusters[-1])} sequences")
print(clusters[-1])
print("Total Number of data points in all the clusters and outliers is:",(total+len(clusters[-1])))

# Plot the clusters
colors = plt.cm.tab20(np.linspace(0, 1, len(set(dbscan.labels_))))
for label in np.unique(dbscan.labels_):
    if label == -1:
        # Plot the outliers in black
        plt.scatter(np.array(range(len(ids)))[dbscan.labels_ == label],
                    np.array(distances[:,0])[dbscan.labels_ == label], c='black',edgecolors=None, s=10, label='Outliers')
    else:
        # Plot each cluster with a different color and label
        cluster_ids = clusters[label]
        plt.scatter(np.array(range(len(ids)))[dbscan.labels_ == label], np.array(distances[:,0])
                    [dbscan.labels_ == label], c=colors[label],edgecolors=None, s=10, 
                    label=f'Cluster {label} ({len(cluster_ids)} sequences)')

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()
