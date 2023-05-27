from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict


DATA_DIR = Path('/hdd/bdhammel/photo_dataset/')
db_path = DATA_DIR/'db.pkl'

parser = argparse.ArgumentParser()
parser.add_argument('--img-path', type=Path)

args = parser.parse_args()


df = pd.read_pickle(db_path)

if args.img_path is not None:
    query_path = str(args.img_path)  # .relative_to(DATA_DIR)
    query = df.vector[df.path == query_path].values[0]

    # calculate cosine similarity of query vector will all vectors in db
    # df['sim'] = df.vector.apply(lambda x: np.dot(x, query) / (np.linalg.norm(x) * np.linalg.norm(query)))

# build a simulatiry matrix for all vectors in db
vectors = np.asarray(df.vector.tolist())
vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
unit_vectors = vectors / vectors_norm
sim = np.dot(unit_vectors, unit_vectors.T)

thresh = np.zeros_like(sim)
thresh[sim > 0.9] = 1

# Get all indices of similar images
indices = np.argwhere(thresh)


class DisjointSet:

    def __init__(self):
        self.roots = {}
        self.groups = defaultdict(set)

    def find(self, x):
        r = self.roots.get(x, None)
        if r is None:
            r = x
            self.roots[x] = x
            self.groups[x].add(x)
        if r != x:
            r = self.find(r)
            self.roots[x] = r
        return r

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.roots[ry] = rx
            self.groups[rx] |= self.groups[ry]
            del self.groups[ry]


ds = DisjointSet()

for edge in indices:
    ds.union(*edge)

# Get all groups of similar images
groups = ds.groups.values()

paths = df.path.tolist()
for group in groups:
    print([paths[i] for i in group])
