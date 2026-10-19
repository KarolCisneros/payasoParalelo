# Corre la rejilla y escribe resultados.csv (cada fila = promedio de REPS iteraciones).
# Antes de medir tiempos, verifica que serial, P1 y P2 (todas las k) producen EXACTAMENTE las mismas etiquetas:
# un speedup solo significa algo si los tres programas calculan lo mismo.
import filecmp, os, shutil, subprocess
from generar_datos import NS, generar

KS = [2, 4, 8, 16, 32, 64]
VCORES = os.cpu_count()
HILOS = sorted({1, max(1, VCORES // 2), VCORES, VCORES * 2})   # sin duplicados si hay pocos nucleos
REPS = int(os.environ.get("REPS", 10))

def correr(n, modo, hilos, k=4, reps=REPS, out=None):
    data = f"data/{n}_data.csv"
    if not os.path.exists(data):
        generar(n)
    out = out or f"data/{n}_results.csv"
    res = subprocess.run(["./dbscan", data, out, modo, str(hilos), str(k), "0.03", "10", str(reps)],
                         check=True, capture_output=True, text=True).stdout
    print(res, end="", flush=True)
    return res

def verificar(n):
    """Misma salida (etiqueta por punto) en serial, P1 y P2 con cada k. Deja la salida serial en data/{n}_results.csv."""
    h = max(HILOS)
    ref = f"data/{n}_ref.csv"
    correr(n, "serial", 1, reps=1, out=ref)
    candidatos = [("p1", 0)] + [("p2", k) for k in KS]
    for modo, k in candidatos:
        tmp = f"data/{n}_chk.csv"
        correr(n, modo, h, k or 4, reps=1, out=tmp)
        assert filecmp.cmp(ref, tmp, shallow=False), f"{modo} (k={k}) difiere de serial para N={n}"
        os.remove(tmp)
    shutil.move(ref, f"data/{n}_results.csv")
    print(f"# N={n}: serial == P1 == P2 (todas las k)", flush=True)

if __name__ == "__main__":
    for n in NS:
        verificar(n)
    with open("resultados.csv", "w") as f:
        f.write("modo,N,hilos,k,t_pre,t_alg,t_total\n")
        for n in NS:
            f.write(correr(n, "serial", 1))
            for h in HILOS:
                f.write(correr(n, "p1", h))
                for k in KS:
                    f.write(correr(n, "p2", h, k))