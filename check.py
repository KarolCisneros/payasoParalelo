# Verifica: serial == P1 == P2 (todos los k) y == sklearn DBSCAN (ruido).
import subprocess, sys
import numpy as np
from sklearn.cluster import DBSCAN
from generar_datos import generar

EPS, MS = 0.03, 10
KS = [2, 4, 8, 16, 32, 64]

def correr(data, modo, hilos=4, k=4):
    subprocess.run(["./dbscan", data, "out_check.csv", modo, str(hilos), str(k), str(EPS), str(MS)],
                   check=True, stdout=subprocess.DEVNULL)
    return np.loadtxt("out_check.csv", delimiter=",")[:, 2]

ok = True
for n in [4000, 20000]:
    data = generar(n)
    base = correr(data, "serial", 1)
    # sklearn cuenta al propio punto como vecino; nosotros no -> min_samples + 1
    pts = np.loadtxt(data, delimiter=",")
    sk = (DBSCAN(eps=EPS, min_samples=MS + 1).fit_predict(pts) != -1).astype(float)
    d = int((base != sk).sum())
    print(f"N={n}: ruido serial={int((base == 0).sum())}, difs vs sklearn={d}")
    ok &= d == 0
    for modo, hilos, k in [("p1", 1, 0), ("p1", 8, 0)] + [("p2", h, k) for k in KS for h in (1, 8)]:
        d = int((correr(data, modo, hilos, k or 4) != base).sum())
        if d:
            print(f"  FALLA {modo} hilos={hilos} k={k}: {d} difs")
        ok &= d == 0
print("OK" if ok else "FALLÓ")
sys.exit(0 if ok else 1)
