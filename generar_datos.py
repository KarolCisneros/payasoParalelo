# Genera data/{N}_data.csv con el mismo make_blobs de DBSCAN_noise.ipynb.
# Uso: python generar_datos.py [N ...]   (sin argumentos: los N del experimento)
import os, sys
import numpy as np
from sklearn.datasets import make_blobs

NS = [20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000]

def generar(n):
    points, _ = make_blobs(n_samples=n, centers=4, cluster_std=0.06, random_state=11, center_box=(0, 1.0))
    points = np.round(np.abs(points[:, ::-1]), 3)
    os.makedirs("data", exist_ok=True)
    path = f"data/{n}_data.csv"
    np.savetxt(path, points, delimiter=",", fmt="%.3f")
    return path

if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]] or NS:
        print(generar(n))
