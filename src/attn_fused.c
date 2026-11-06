/*
 * attn_fused.c: Q3c, the three steps fused row by row.
 *
 * Each row of Q walks over K and V in blocks of BC and keeps a running max
 * m, sum l and output o (online softmax). When the max goes up, l and o are
 * rescaled by exp(m_old - m_new). S is never stored. The inner loops are
 * the same stride-1 loops as in attn_loops.c.
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
#define BC 256

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *Kt = malloc((size_t)N * d * sizeof(float));
    float *s = malloc(BC * sizeof(float));
    float *o = malloc((size_t)d * sizeof(float));
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kt[(size_t)k * N + j] = K[(size_t)j * d + k];

    for (int i = 0; i < N; i++) {
        const float *q = Q + (size_t)i * d;
        float m = -INFINITY, l = 0.0f;
        for (int k = 0; k < d; k++)
            o[k] = 0.0f;

        for (int j0 = 0; j0 < N; j0 += BC) {
            const int bc = N - j0 < BC ? N - j0 : BC;

            /* scores for this block: s = (q . K[j0..j0+bc)) * scale */
            for (int j = 0; j < bc; j++)
                s[j] = 0.0f;
            for (int k = 0; k < d; k++) {
                const float qk = q[k] * scale;
                const float *restrict kt = Kt + (size_t)k * N + j0;
                float *restrict sb = s;
                for (int j = 0; j < bc; j++)
                    sb[j] += qk * kt[j];
            }

            /* online softmax update */
            float mb = s[0];
            for (int j = 1; j < bc; j++)
                mb = s[j] > mb ? s[j] : mb;
            const float mnew = mb > m ? mb : m;
            const float a = expf(m - mnew); /* 0 on the first block */
            float lb = 0.0f;
            for (int j = 0; j < bc; j++) {
                s[j] = expf(s[j] - mnew);
                lb += s[j];
            }
            l = a * l + lb;
            m = mnew;

            /* o = a*o + sum_j p_j V[j] */
            float *restrict ov = o;
            for (int k = 0; k < d; k++)
                ov[k] *= a;
            for (int j = 0; j < bc; j++) {
                const float p = s[j];
                const float *restrict v = V + (size_t)(j0 + j) * d;
                for (int k = 0; k < d; k++)
                    ov[k] += p * v[k];
            }
        }

        const float inv = 1.0f / l;
        for (int k = 0; k < d; k++)
            O[(size_t)i * d + k] = o[k] * inv;
    }

    free(o);
    free(s);
    free(Kt);
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
