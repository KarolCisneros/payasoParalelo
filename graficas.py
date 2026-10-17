# Gráficas de speedup (T_serial / T_total) a partir de resultados.csv -> graficas/*.png
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = np.genfromtxt("resultados.csv", delimiter=",", names=True, dtype=None, encoding=None)
os.makedirs("graficas", exist_ok=True)
NS = sorted(set(r["N"]))
HILOS = sorted(set(r["hilos"]))
KS = sorted(set(r["k"][r["modo"] == "p2"]))
serial = {row["N"]: row["t_total"] for row in r[r["modo"] == "serial"]}

def t(modo, n, h, k=0):
    m = (r["modo"] == modo) & (r["N"] == n) & (r["hilos"] == h) & (r["k"] == k)
    return r["t_total"][m][0]

def guardar(nombre, titulo, xlabel, ylabel="Speedup"):
    plt.title(titulo); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"graficas/{nombre}.png", dpi=150); plt.close()

for h in HILOS:
    plt.plot(NS, [serial[n] / t("p1", n, h) for n in NS], "o-", label=f"{h} hilos")
guardar("speedup_p1_vs_N", "P1: speedup vs N", "N")

for h in HILOS:
    for k in KS:
        plt.plot(NS, [serial[n] / t("p2", n, h, k) for n in NS], "o-", label=f"k={k}")
    guardar(f"speedup_p2_vs_N_{h}hilos", f"P2 ({h} hilos): speedup vs N", "N")

nmax = NS[-1]
for h in HILOS:
    plt.plot(KS, [serial[nmax] / t("p2", nmax, h, k) for k in KS], "o-", label=f"{h} hilos")
plt.xscale("log", base=2)
guardar("speedup_p2_vs_k", f"P2: speedup vs k (N={nmax})", "k")

plt.plot(NS, [serial[n] for n in NS], "o-", label="serial")
plt.plot(NS, [min(t("p1", n, h) for h in HILOS) for n in NS], "o-", label="mejor P1")
plt.plot(NS, [min(t("p2", n, h, k) for h in HILOS for k in KS) for n in NS], "o-", label="mejor P2")
guardar("tiempos_vs_N", "Tiempo total vs N", "N", "segundos")

res = np.loadtxt(f"data/{nmax}_results.csv", delimiter=",")
plt.scatter(res[:, 0], res[:, 1], c=res[:, 2], s=1)
plt.title(f"Ruido detectado (N={nmax})"); plt.axis("off")
plt.savefig("graficas/ruido.png", dpi=150); plt.close()
print("\n".join(sorted(os.listdir("graficas"))))
