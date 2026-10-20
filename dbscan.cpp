// DBSCAN para detección de ruido: versión serial, P1 (espacio indivisible) y P2 (k regiones con halo).
// Uso: ./dbscan <entrada.csv> <salida.csv> <serial|p1|p2> <hilos> [k=4] [eps=0.03] [min_samples=10] [reps=1]
// Imprime: modo,N,hilos,k,t_pre,t_alg,t_total (segundos, promedio de reps, sin E/S)
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;

enum { RUIDO_DEFINITIVO = 0, CORE = 1, CORE_SEGUNDO_GRADO = 2, POSIBLE_RUIDO = 3 };

// double y no float: coordenadas de 3 decimales con eps=0.03 dan muchas distancias exactamente en eps
static vector<double> x, y;
static vector<unsigned char> es_core, estado;   // unsigned char, NO vector<bool> (bits compartidos)
static int N, min_samples;
static double eps, eps2;

static inline double dist2(int a, int b) {
    double dx = x[a] - x[b], dy = y[a] - y[b];
    return dx * dx + dy * dy;
}

// ---------------- Serial: P1 sin pragmas ----------------
static void dbscan_serial() {
    for (int i = 0; i < N; i++) {
        int vecinos = 0;
        for (int j = 0; j < N; j++)
            if (i != j && dist2(i, j) <= eps2 && ++vecinos >= min_samples) break;
        es_core[i] = vecinos >= min_samples;
        estado[i] = es_core[i] ? CORE : POSIBLE_RUIDO;
    }
    for (int i = 0; i < N; i++) {
        if (estado[i] != POSIBLE_RUIDO) continue;
        int alcanzable = 0;
        for (int j = 0; j < N; j++)
            if (es_core[j] && dist2(i, j) <= eps2) { alcanzable = 1; break; }
        estado[i] = alcanzable ? CORE_SEGUNDO_GRADO : RUIDO_DEFINITIVO;
    }
}

// ---------------- P1: se reparten los puntos entre hilos ----------------
static void dbscan_p1() {
    // Fase 1: ¿quién es core? Cada hilo escribe solo es_core[i]/estado[i] de su i.
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; i++) {
        int vecinos = 0;
        for (int j = 0; j < N; j++)
            if (i != j && dist2(i, j) <= eps2 && ++vecinos >= min_samples) break;
        es_core[i] = vecinos >= min_samples;
        estado[i] = es_core[i] ? CORE : POSIBLE_RUIDO;
    }
    // <- barrera implícita: es_core[] completo antes de la Fase 2
    // Fase 2: ¿el posible ruido tiene algún vecino core? es_core[] solo se lee.
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; i++) {
        if (estado[i] != POSIBLE_RUIDO) continue;
        int alcanzable = 0;
        for (int j = 0; j < N; j++)
            if (es_core[j] && dist2(i, j) <= eps2) { alcanzable = 1; break; }
        estado[i] = alcanzable ? CORE_SEGUNDO_GRADO : RUIDO_DEFINITIVO;
    }
}

// ---------------- P2: k regiones (M internos + B halo de solo lectura) ----------------
struct Region { vector<int> M, B; };

static vector<Region> construir_regiones(int k) {
    int gx = 1 << ((__builtin_ctz(k) + 1) / 2), gy = k / gx;   // 2x1, 2x2, 4x2, 4x4, 8x4, 8x8
    double xmin = *min_element(x.begin(), x.end()), xmax = *max_element(x.begin(), x.end());
    double ymin = *min_element(y.begin(), y.end()), ymax = *max_element(y.begin(), y.end());
    double cw = (xmax - xmin) / gx, ch = (ymax - ymin) / gy;
    double eh = eps * (1 + 1e-9);   // halo un pelo más ancho: el redondeo no deja fuera vecinos a exactamente eps
    vector<Region> reg(k);
    for (int i = 0; i < N; i++) {
        int cx = min((int)((x[i] - xmin) / cw), gx - 1);
        int cy = min((int)((y[i] - ymin) / ch), gy - 1);
        int dueno = cy * gx + cx;
        reg[dueno].M.push_back(i);
        // celdas que toca el cuadro [x-eps, x+eps] x [y-eps, y+eps]
        int cx0 = max((int)floor((x[i] - eh - xmin) / cw), 0), cx1 = min((int)((x[i] + eh - xmin) / cw), gx - 1);
        int cy0 = max((int)floor((y[i] - eh - ymin) / ch), 0), cy1 = min((int)((y[i] + eh - ymin) / ch), gy - 1);
        for (int ry = cy0; ry <= cy1; ry++)
            for (int rx = cx0; rx <= cx1; rx++) {
                int r = ry * gx + rx;
                if (r != dueno) reg[r].B.push_back(i);   // i es halo de la región r
            }
    }
    return reg;
}

static void dbscan_p2(vector<Region> &reg) {
    int k = reg.size();
    // Fase 1: cada región etiqueta solo sus puntos internos M (cada punto está en una sola M).
    #pragma omp parallel for schedule(dynamic, 1)
    for (int r = 0; r < k; r++) {
        const Region &R = reg[r];
        for (int i : R.M) {
            int vecinos = 0;
            for (size_t b = 0; b < R.M.size() && vecinos < min_samples; b++)
                if (i != R.M[b] && dist2(i, R.M[b]) <= eps2) vecinos++;
            for (size_t b = 0; b < R.B.size() && vecinos < min_samples; b++)
                if (dist2(i, R.B[b]) <= eps2) vecinos++;
            es_core[i] = vecinos >= min_samples;
            estado[i] = es_core[i] ? CORE : POSIBLE_RUIDO;
        }
    }
    // <- barrera implícita: es_core[] de TODAS las regiones (incluido el halo) ya es definitivo
    // Fase 2: resolver ruido leyendo es_core[] global.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int r = 0; r < k; r++) {
        const Region &R = reg[r];
        for (int i : R.M) {
            if (estado[i] != POSIBLE_RUIDO) continue;
            int alcanzable = 0;
            for (size_t b = 0; b < R.M.size() && !alcanzable; b++)
                if (es_core[R.M[b]] && dist2(i, R.M[b]) <= eps2) alcanzable = 1;
            for (size_t b = 0; b < R.B.size() && !alcanzable; b++)
                if (es_core[R.B[b]] && dist2(i, R.B[b]) <= eps2) alcanzable = 1;
            estado[i] = alcanzable ? CORE_SEGUNDO_GRADO : RUIDO_DEFINITIVO;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "uso: %s <entrada.csv> <salida.csv> <serial|p1|p2> <hilos> [k=4] [eps=0.03] [min_samples=10] [reps=1]\n", argv[0]);
        return 1;
    }
    const char *modo = argv[3];
    int hilos = atoi(argv[4]);
    int k = argc > 5 ? atoi(argv[5]) : 4;
    eps = argc > 6 ? atof(argv[6]) : 0.03;
    min_samples = argc > 7 ? atoi(argv[7]) : 10;
    int reps = argc > 8 ? atoi(argv[8]) : 1;
    eps2 = eps * eps;
    bool serial = !strcmp(modo, "serial"), p1 = !strcmp(modo, "p1"), p2 = !strcmp(modo, "p2");
    if (!(serial || p1 || p2) || hilos < 1 || reps < 1 || (p2 && (k < 2 || (k & (k - 1))))) {
        fprintf(stderr, "argumentos inválidos (k debe ser potencia de 2 >= 2)\n");
        return 1;
    }
    omp_set_num_threads(hilos);

    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    double a, b;
    while (fscanf(f, " %lf , %lf", &a, &b) == 2) { x.push_back(a); y.push_back(b); }
    fclose(f);
    N = x.size();
    es_core.assign(N, 0);
    estado.assign(N, 0);

    double t_pre = 0, t_alg = 0;
    for (int it = 0; it < reps; it++) {
        double t0 = omp_get_wtime();
        vector<Region> reg;
        if (p2) reg = construir_regiones(k);   // preprocesamiento serial, cuenta en el costo de P2
        double t1 = omp_get_wtime();
        if (serial) dbscan_serial();
        else if (p1) dbscan_p1();
        else dbscan_p2(reg);
        double t2 = omp_get_wtime();
        t_pre += t1 - t0;
        t_alg += t2 - t1;
    }
    t_pre /= reps;
    t_alg /= reps;
    printf("%s,%d,%d,%d,%.6f,%.6f,%.6f\n", modo, N, hilos, p2 ? k : 0, t_pre, t_alg, t_pre + t_alg);

    // Salida para el notebook: x,y,etiqueta (1 = core de 1er o 2º grado, 0 = ruido)
    FILE *o = fopen(argv[2], "w");
    if (!o) { perror(argv[2]); return 1; }
    for (int i = 0; i < N; i++) fprintf(o, "%.3f,%.3f,%d\n", x[i], y[i], estado[i] != RUIDO_DEFINITIVO);
    fclose(o);
    return 0;
}
