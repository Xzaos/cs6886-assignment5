/*
 * attn_tiled.c: Q3b, cache and register tiling.
 *
 * S is still stored. Both products work on 4 x 16 output tiles whose
 * accumulators stay in registers, and K and V are packed into 16-wide
 * panels so the kernel reads them contiguously. The j loop is blocked by
 * 256 so the panels stay in cache while the rows of Q use them.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
#define MR 4
#define NR 16
#define JB 256

/* acc[MR][NR] (row stride N) = Qs[i..i+MR) . Kp micro-panel, over k < d */
static inline void kernel_qk(const float *restrict qs, int d, const float *restrict kp,
                             float *restrict s, int N)
{
    float acc[MR][NR] = {{0}};
    for (int k = 0; k < d; k++) {
        const float *restrict b = kp + (size_t)k * NR;
        for (int r = 0; r < MR; r++) {
            const float a = qs[(size_t)r * d + k];
            for (int c = 0; c < NR; c++)
                acc[r][c] += a * b[c];
        }
    }
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++)
            s[(size_t)r * N + c] = acc[r][c];
}

/* O[i..i+MR)[c0..c0+NR) += P[i..i+MR)[j0..j1) . Vp slice */
static inline void kernel_pv(const float *restrict p, int N, int len, const float *restrict vp,
                             float *restrict o, int d)
{
    float acc[MR][NR];
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++)
            acc[r][c] = o[(size_t)r * d + c];
    for (int j = 0; j < len; j++) {
        const float *restrict b = vp + (size_t)j * NR;
        for (int r = 0; r < MR; r++) {
            const float a = p[(size_t)r * N + j];
            for (int c = 0; c < NR; c++)
                acc[r][c] += a * b[c];
        }
    }
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++)
            o[(size_t)r * d + c] = acc[r][c];
}

static void softmax_rows(float *S, int N)
{
    for (int i = 0; i < N; i++) {
        float *restrict row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            max = row[j] > max ? row[j] : max;
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < N; j++)
            row[j] *= inv;
    }
}

/* Untiled fallback for shapes the tiles do not divide (not used by the given inputs). */
static void attention_simple(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *S = malloc((size_t)N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < d; k++)
                acc += Q[(size_t)i * d + k] * K[(size_t)j * d + k];
            S[(size_t)i * N + j] = acc * scale;
        }
    softmax_rows(S, N);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++)
            O[(size_t)i * d + k] = 0.0f;
        for (int j = 0; j < N; j++)
            for (int k = 0; k < d; k++)
                O[(size_t)i * d + k] += S[(size_t)i * N + j] * V[(size_t)j * d + k];
    }
    free(S);
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    if (N % JB || N % MR || d % NR) {
        attention_simple(Q, K, V, O, N, d);
        return;
    }
    const float scale = 1.0f / sqrtf((float)d);
    float *S = aligned_alloc(64, (size_t)N * N * sizeof(float));
    float *Qs = aligned_alloc(64, (size_t)N * d * sizeof(float));
    float *Kp = aligned_alloc(64, (size_t)N * d * sizeof(float));
    float *Vp = aligned_alloc(64, (size_t)N * d * sizeof(float));

    /* Qs = Q * scale, Kp[jp][k][c] = K[jp*16 + c][k], Vp[cp][j][c] = V[j][cp*16 + c] */
    for (size_t x = 0; x < (size_t)N * d; x++)
        Qs[x] = Q[x] * scale;
    for (int jp = 0; jp < N / NR; jp++)
        for (int k = 0; k < d; k++)
            for (int c = 0; c < NR; c++)
                Kp[((size_t)jp * d + k) * NR + c] = K[(size_t)(jp * NR + c) * d + k];
    for (int cp = 0; cp < d / NR; cp++)
        for (int j = 0; j < N; j++)
            for (int c = 0; c < NR; c++)
                Vp[((size_t)cp * N + j) * NR + c] = V[(size_t)j * d + cp * NR + c];

    /* 1. S = Qs K^T: j-blocks of JB columns, all rows of Q, 4 x 16 tiles */
    for (int j0 = 0; j0 < N; j0 += JB)
        for (int i = 0; i < N; i += MR)
            for (int j = j0; j < j0 + JB; j += NR)
                kernel_qk(Qs + (size_t)i * d, d, Kp + (size_t)(j / NR) * d * NR,
                          S + (size_t)i * N + j, N);

    /* 2. softmax */
    softmax_rows(S, N);

    /* 3. O = S V: j-blocks of JB rows of V, 4 x 16 tiles of O */
    memset(O, 0, (size_t)N * d * sizeof(float));
    for (int j0 = 0; j0 < N; j0 += JB)
        for (int i = 0; i < N; i += MR)
            for (int cp = 0; cp < d / NR; cp++)
                kernel_pv(S + (size_t)i * N + j0, N, JB, Vp + ((size_t)cp * N + j0) * NR,
                          O + (size_t)i * d + cp * NR, d);

    free(Vp);
    free(Kp);
    free(Qs);
    free(S);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
