# Corre la rejilla del PDF y escribe resultados.csv (cada fila = promedio de REPS iteraciones).
import os, subprocess
from generar_datos import NS, generar

KS = [2, 4, 8, 16, 32, 64]
VCORES = os.cpu_count()
HILOS = [1, VCORES // 2, VCORES, VCORES * 2]
REPS = 10

def correr(n, modo, hilos, k=4):
    data = f"data/{n}_data.csv"
    if not os.path.exists(data):
        generar(n)
    out = subprocess.run(["./dbscan", data, f"data/{n}_results.csv", modo, str(hilos), str(k), "0.03", "10", str(REPS)],
                         check=True, capture_output=True, text=True).stdout
    print(out, end="", flush=True)
    return out

with open("resultados.csv", "w") as f:
    f.write("modo,N,hilos,k,t_pre,t_alg,t_total\n")
    for n in NS:
        f.write(correr(n, "serial", 1))
        for h in HILOS:
            f.write(correr(n, "p1", h))
            for k in KS:
                f.write(correr(n, "p2", h, k))
