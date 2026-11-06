/*
 * attn_best.c: Q3d, fastest single-threaded version.
 *
 * Fused like attn_fused.c, but on blocks of 32 rows of Q and 64 rows of K/V,
 * so each K/V block is reused by 32 rows while it is in L1. Both products use
 * 4 x 16 register tiles written with AVX2 intrinsics. Q is pre-scaled by
 * log2(e)/sqrt(d) so the softmax can use a vectorised exp2 instead of expf.
 * Falls back to a plain fused loop without AVX2 or if the shape doesn't fit
 * the tiles.
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
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HAVE_AVX2 1
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
#define BR 32 /* rows of Q per block   */
#define BC 64 /* rows of K, V per block */
#define MR 4  /* micro-tile rows       */
#define NR 16 /* micro-tile columns    */

/* Plain fused version, for shapes the tiles do not divide or CPUs without AVX2. */
static void attention_fallback(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *o = malloc((size_t)d * sizeof(float));
    for (int i = 0; i < N; i++) {
        float m = -INFINITY, l = 0.0f;
        for (int k = 0; k < d; k++)
            o[k] = 0.0f;
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < d; k++)
                s += Q[(size_t)i * d + k] * K[(size_t)j * d + k];
            s *= scale;
            const float mnew = s > m ? s : m;
            const float a = expf(m - mnew), p = expf(s - mnew);
            l = a * l + p;
            for (int k = 0; k < d; k++)
                o[k] = a * o[k] + p * V[(size_t)j * d + k];
            m = mnew;
        }
        for (int k = 0; k < d; k++)
            O[(size_t)i * d + k] = o[k] / l;
    }
    free(o);
}

#ifdef HAVE_AVX2
static inline __m256 exp2v(__m256 x)
{
    x = _mm256_max_ps(x, _mm256_set1_ps(-126.0f));
    const __m256 n = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 f = _mm256_sub_ps(x, n);
    __m256 p = _mm256_set1_ps(1.5403530e-4f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.3333558e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6181291e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5504109e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022651e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

static inline float hmax(__m256 v)
{
    __m128 x = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_max_ps(x, _mm_movehl_ps(x, x));
    x = _mm_max_ss(x, _mm_shuffle_ps(x, x, 1));
    return _mm_cvtss_f32(x);
}

static inline float hsum(__m256 v)
{
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_shuffle_ps(x, x, 1));
    return _mm_cvtss_f32(x);
}

/* s[r][0..16) (row stride BC) = sum_k q[r][k] * kp[k][0..16), r < 4 */
static inline void kernel_qk(const float *restrict q, int d, const float *restrict kp, float *restrict s)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00;
    __m256 c20 = c00, c21 = c00, c30 = c00, c31 = c00;
    for (int k = 0; k < d; k++) {
        const __m256 b0 = _mm256_load_ps(kp + k * NR), b1 = _mm256_load_ps(kp + k * NR + 8);
        __m256 a = _mm256_broadcast_ss(q + k);
        c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(q + d + k);
        c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(q + 2 * d + k);
        c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(q + 3 * d + k);
        c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
    }
    _mm256_store_ps(s, c00);          _mm256_store_ps(s + 8, c01);
    _mm256_store_ps(s + BC, c10);     _mm256_store_ps(s + BC + 8, c11);
    _mm256_store_ps(s + 2 * BC, c20); _mm256_store_ps(s + 2 * BC + 8, c21);
    _mm256_store_ps(s + 3 * BC, c30); _mm256_store_ps(s + 3 * BC + 8, c31);
}

/* o[r][0..16) (row stride d) = alpha[r] * o[r] + sum_{j<BC} p[r][j] * v[j][0..16), r < 4 */
static inline void kernel_pv(const float *restrict p, const float *restrict alpha, const float *restrict v,
                             int d, float *restrict o)
{
    __m256 a0 = _mm256_broadcast_ss(alpha), a1 = _mm256_broadcast_ss(alpha + 1);
    __m256 a2 = _mm256_broadcast_ss(alpha + 2), a3 = _mm256_broadcast_ss(alpha + 3);
    __m256 c00 = _mm256_mul_ps(a0, _mm256_load_ps(o)), c01 = _mm256_mul_ps(a0, _mm256_load_ps(o + 8));
    __m256 c10 = _mm256_mul_ps(a1, _mm256_load_ps(o + d)), c11 = _mm256_mul_ps(a1, _mm256_load_ps(o + d + 8));
    __m256 c20 = _mm256_mul_ps(a2, _mm256_load_ps(o + 2 * d)), c21 = _mm256_mul_ps(a2, _mm256_load_ps(o + 2 * d + 8));
    __m256 c30 = _mm256_mul_ps(a3, _mm256_load_ps(o + 3 * d)), c31 = _mm256_mul_ps(a3, _mm256_load_ps(o + 3 * d + 8));
    for (int j = 0; j < BC; j++) {
        const __m256 b0 = _mm256_loadu_ps(v + (size_t)j * d), b1 = _mm256_loadu_ps(v + (size_t)j * d + 8);
        __m256 a = _mm256_broadcast_ss(p + j);
        c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(p + BC + j);
        c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(p + 2 * BC + j);
        c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(p + 3 * BC + j);
        c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
    }
    _mm256_store_ps(o, c00);         _mm256_store_ps(o + 8, c01);
    _mm256_store_ps(o + d, c10);     _mm256_store_ps(o + d + 8, c11);
    _mm256_store_ps(o + 2 * d, c20); _mm256_store_ps(o + 2 * d + 8, c21);
    _mm256_store_ps(o + 3 * d, c30); _mm256_store_ps(o + 3 * d + 8, c31);
}

/* Rows [i0, i0+BR) of O. Scratch: s (BR*BC), acc (BR*d), m, l, alpha (BR each). */
static void attn_rows(const float *Qs, const float *Kp, const float *V, float *O, int N, int d, int i0,
                      float *s, float *acc, float *m, float *l, float *alpha)
{
    const float *q = Qs + (size_t)i0 * d;
    memset(acc, 0, (size_t)BR * d * sizeof(float));
    for (int r = 0; r < BR; r++) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }
    for (int j0 = 0; j0 < N; j0 += BC) {
        /* score tile s = q K[j0..j0+BC)^T, already scaled and in base 2 */
        for (int r = 0; r < BR; r += MR)
            for (int jp = 0; jp < BC; jp += NR)
                kernel_qk(q + (size_t)r * d, d, Kp + (size_t)(j0 + jp) * d, s + r * BC + jp);

        /* online softmax: s <- 2^(s - m_new), alpha = 2^(m_old - m_new) */
        for (int r = 0; r < BR; r++) {
            float *sr = s + r * BC;
            __m256 mx = _mm256_load_ps(sr);
            for (int j = 8; j < BC; j += 8)
                mx = _mm256_max_ps(mx, _mm256_load_ps(sr + j));
            const float mb = hmax(mx);
            const float mnew = mb > m[r] ? mb : m[r];
            const __m256 mv = _mm256_set1_ps(mnew);
            __m256 sum = _mm256_setzero_ps();
            for (int j = 0; j < BC; j += 8) {
                const __m256 e = exp2v(_mm256_sub_ps(_mm256_load_ps(sr + j), mv));
                _mm256_store_ps(sr + j, e);
                sum = _mm256_add_ps(sum, e);
            }
            alpha[r] = exp2f(m[r] - mnew);
            l[r] = alpha[r] * l[r] + hsum(sum);
            m[r] = mnew;
        }

        /* acc = alpha * acc + P V[j0..j0+BC) */
        const float *vb = V + (size_t)j0 * d;
        for (int r = 0; r < BR; r += MR)
            for (int c = 0; c < d; c += NR)
                kernel_pv(s + r * BC, alpha + r, vb + c, d, acc + (size_t)r * d + c);
    }
    for (int r = 0; r < BR; r++) {
        const __m256 inv = _mm256_set1_ps(1.0f / l[r]);
        for (int c = 0; c < d; c += 8)
            _mm256_storeu_ps(O + (size_t)(i0 + r) * d + c,
                             _mm256_mul_ps(inv, _mm256_load_ps(acc + (size_t)r * d + c)));
    }
}

/* Qs = Q * log2(e)/sqrt(d);  Kp[j/16][k][j%16] = K[j][k] */
static void prepare(const float *Q, const float *K, float *Qs, float *Kp, int N, int d)
{
    const float qscale = 1.4426950408889634f / sqrtf((float)d);
    for (size_t x = 0; x < (size_t)N * d; x++)
        Qs[x] = Q[x] * qscale;
    for (int jp = 0; jp < N / NR; jp++)
        for (int k = 0; k < d; k++)
            for (int c = 0; c < NR; c++)
                Kp[((size_t)jp * d + k) * NR + c] = K[(size_t)(jp * NR + c) * d + k];
}
#endif

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
#ifdef HAVE_AVX2
    if (N % BR == 0 && N % BC == 0 && d % NR == 0) {
        float *Qs = aligned_alloc(64, (size_t)N * d * sizeof(float));
        float *Kp = aligned_alloc(64, (size_t)N * d * sizeof(float));
        float *s = aligned_alloc(64, BR * BC * sizeof(float));
        float *acc = aligned_alloc(64, (size_t)BR * d * sizeof(float));
        float m[BR], l[BR], alpha[BR];
        prepare(Q, K, Qs, Kp, N, d);
        for (int i0 = 0; i0 < N; i0 += BR)
            attn_rows(Qs, Kp, V, O, N, d, i0, s, acc, m, l, alpha);
        free(acc);
        free(s);
        free(Kp);
        free(Qs);
        return;
    }
#endif
    attention_fallback(Q, K, V, O, N, d);
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
