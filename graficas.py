# Gráficas de speedup a partir de resultados.csv -> graficas/*.png
#
# Idea central: el speedup de P2 sobre el serial mezcla DOS efectos distintos,
#   S_total = T_serial / T_P2(k,h) = [T_serial / T_P2(k,1)] x [T_P2(k,1) / T_P2(k,h)]
#                                     ganancia ALGORÍTMICA      ganancia por PARALELISMO
# El serial compara cada punto contra los N puntos (fuerza bruta); P2 solo contra los de su región + halo
# (~N/k puntos). Eso es menos trabajo, no más hilos. Amdahl y el tope "speedup <= hilos" aplican solo al segundo factor.
# Por eso cada gráfica declara su línea base en el título.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

r = np.genfromtxt("resultados.csv", delimiter=",", names=True, dtype=None, encoding=None)
os.makedirs("graficas", exist_ok=True)
NS = sorted(set(r["N"]))
HILOS = sorted(set(r["hilos"]))
KS = sorted(set(r["k"][r["modo"] == "p2"]))
V = HILOS[-2] if len(HILOS) >= 3 else HILOS[-1]      # hilos lógicos de la máquina (la rejilla es 1, V/2, V, 2V)
NMAX = NS[-1]

GRIS, NARANJA, VERDE = "#555555", "#d9731a", "#2a9d5c"
col_k = dict(zip(KS, plt.cm.viridis(np.linspace(0.05, 0.9, len(KS)))))
col_h = dict(zip(HILOS, ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]))

def t(modo, n, h, k=0):
    m = (r["modo"] == modo) & (r["N"] == n) & (r["hilos"] == h) & (r["k"] == k)
    return r["t_total"][m][0]

def pre(n, h, k):
    m = (r["modo"] == "p2") & (r["N"] == n) & (r["hilos"] == h) & (r["k"] == k)
    return r["t_pre"][m][0] / r["t_total"][m][0]

ts = {n: t("serial", n, 1) for n in NS}

def estilo(ax, titulo, sub, ylabel="Speedup"):
    ax.set_title(f"{titulo}\n{sub}" if sub else titulo, fontsize=9.5)
    ax.set_xlabel("N (número de puntos)"); ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.grid(alpha=.3)

def guardar(fig, nombre):
    fig.tight_layout(); fig.savefig(f"graficas/{nombre}.png", dpi=150); plt.close(fig)

# 1. Serial: es la línea base (speedup = 1 por definición), así que se grafica su tiempo.
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.plot(NS, [ts[n] for n in NS], "o-", color=GRIS, label="serial (fuerza bruta, 1 hilo)")
estilo(ax, "1. Serial: la línea base de todo lo demás",
       "Se grafica tiempo, no speedup (su speedup contra sí mismo es 1)", "Tiempo total (s)")
ax.legend()
guardar(fig, "1_serial")

# 2. P1: mismo algoritmo que serial, solo repartido entre hilos -> comparación limpia.
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for h in HILOS:
    ax.plot(NS, [ts[n] / t("p1", n, h) for n in NS], "o-", color=col_h[h], label=f"{h} hilos")
ax.axhline(V, ls="--", color="gray", lw=1)
ax.text(NS[-1], V, f"tope teórico: {V} hilos lógicos ", ha="right", va="bottom", fontsize=8, color="gray")
estilo(ax, "2. P1: speedup puro por paralelismo",
       "Speedup = T_serial(N) / T_P1(N, hilos)\n(mismo algoritmo, mismo trabajo)")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
guardar(fig, "2_p1")

# 3a. P2 con UN hilo: no hay paralelismo, todo lo que supere 1x es trabajo que el serial hacía de más.
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for k in KS:
    ax.plot(NS, [ts[n] / t("p2", n, 1, k) for n in NS], "o-", color=col_k[k], label=f"k={k}")
ax.axhline(1, ls="--", color="gray", lw=1)
estilo(ax, "3a. P2 con 1 hilo: ganancia ALGORÍTMICA (sin paralelismo)",
       "Speedup = T_serial(N) / T_P2(N, k, 1 hilo)\n(incluye el preprocesamiento de regiones)")
ax.legend(title="regiones", loc="center left", bbox_to_anchor=(1.01, 0.5))
guardar(fig, "3a_p2_algoritmico")

# 3b. P2 contra sí mismo (misma k, 1 hilo): lo único que cambia es el número de hilos.
hs = [h for h in HILOS if h > 1]
fig, axs = plt.subplots(1, len(hs), figsize=(4.6 * len(hs), 4.6), sharey=True, squeeze=False)
for ax, h in zip(axs[0], hs):
    for k in KS:
        ax.plot(NS, [t("p2", n, 1, k) / t("p2", n, h, k) for n in NS], "o-", color=col_k[k], label=f"k={k}")
    ax.axhline(h, ls="--", color="gray", lw=1)
    estilo(ax, f"{h} hilos", "", "Speedup paralelo puro" if ax is axs[0][0] else "")
axs[0][0].legend(title="regiones", ncol=2, fontsize=8)
fig.suptitle("3b. P2: ganancia por PARALELISMO\nSpeedup = T_P2(N, k, 1 hilo) / T_P2(N, k, hilos)   (línea punteada: ideal = hilos)",
             fontsize=9.5)
fig.tight_layout(); fig.savefig("graficas/3b_p2_paralelo.png", dpi=150); plt.close(fig)

# 4. Comparación justa: mismas máquina e hilos (V) para P1 y P2; P2 con UNA k (la mejor a N máximo).
kb = min(KS, key=lambda k: t("p2", NMAX, V, k))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=False)
a1.plot(NS, [ts[n] / t("p1", n, V) for n in NS], "o-", color=NARANJA, label=f"P1, {V} hilos")
a1.plot(NS, [ts[n] / t("p2", n, 1, kb) for n in NS], "s--", color=VERDE, label=f"P2 (k={kb}), 1 hilo: solo algoritmo")
a1.plot(NS, [ts[n] / t("p2", n, V, kb) for n in NS], "o-", color=VERDE, label=f"P2 (k={kb}), {V} hilos")
a1.axhline(1, color=GRIS, lw=1.2, label="serial (fuerza bruta)")
a1.axhline(V, ls=":", color="gray", lw=1)
a1.text(NS[-1], V * 1.25, f"tope por paralelismo: {V} ", ha="right", va="bottom", fontsize=8, color="gray")
estilo(a1, "4a. Contra el serial de fuerza bruta",
       "Speedup = T_serial / T_método\n(lo que supera la línea punteada no es paralelismo)")
a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))

mejor = {n: t("p2", n, 1, kb) for n in NS}   # mejor secuencial disponible: P2 con 1 hilo
a2.plot(NS, [mejor[n] / t("p1", n, V) for n in NS], "o-", color=NARANJA, label=f"P1, {V} hilos")
a2.plot(NS, [mejor[n] / t("p2", n, V, kb) for n in NS], "o-", color=VERDE, label=f"P2 (k={kb}), {V} hilos")
a2.axhline(1, color=GRIS, lw=1.2, label=f"mejor secuencial = P2 (k={kb}) con 1 hilo")
a2.axhline(V, ls=":", color="gray", lw=1)
estilo(a2, "4b. Contra el mejor secuencial",
       f"Speedup = T_P2(k={kb}, 1 hilo) / T_método\n(este sí respeta el tope de {V} hilos)")
a2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))
guardar(fig, "4_comparacion_justa")

# Ruido detectado (sin cambios)
res = np.loadtxt(f"data/{NMAX}_results.csv", delimiter=",")
plt.scatter(res[:, 0], res[:, 1], c=res[:, 2], s=1)
plt.title(f"Ruido detectado (N={NMAX})"); plt.axis("off")
plt.savefig("graficas/ruido.png", dpi=150); plt.close()

# Tabla de descomposición a N máximo y V hilos: S_total = S_alg x S_par
print(f"\nN={NMAX}, {V} hilos. S_total = S_alg x S_par; pre = fracción serial (construir regiones) del tiempo de P2")
print(f"{'k':>4} {'S_total':>8} {'S_alg':>7} {'S_par':>7} {'S_par/min(k,h)':>15} {'pre':>6}")
for k in KS:
    sa, sp = ts[NMAX] / t("p2", NMAX, 1, k), t("p2", NMAX, 1, k) / t("p2", NMAX, V, k)
    print(f"{k:>4} {sa*sp:>8.2f} {sa:>7.2f} {sp:>7.2f} {sp/min(k, V):>15.2f} {pre(NMAX, V, k):>6.0%}")
print(f"P1 con {V} hilos: {ts[NMAX] / t('p1', NMAX, V):.2f}x\n")
print("\n".join(sorted(os.listdir("graficas"))))