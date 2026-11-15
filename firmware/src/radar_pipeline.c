/*
 * Privacy-safe radar gesture recognition: on-device pipeline (see radar_pipeline.h).
 *
 * The INT8 kernels reproduce the TensorFlow Lite graph
 *   CONV(+ReLU) -> MAXPOOL -> CONV(+ReLU) -> MAXPOOL -> CONV(+ReLU) -> MEAN -> FC
 * but fuse each convolution with the operation that follows it, so the large
 * 90x32x16 output of the first convolution is never stored:
 *   conv1+pool1, conv2+pool2, conv3+global-average-pool.
 */
#include "radar_pipeline.h"
#include "model_data.h"

/* No libm needed: sqrt maps to the FPU instruction, floor and the twiddle
 * table are computed here. */
#define SQRTF(x) __builtin_sqrtf(x)

#define N_DFT   RP_BINS
#define K_HALF  ((N_DFT + 1) / 2)          /* bins 0..94 kept by the Hilbert filter (N odd) */

static float cos_t[N_DFT], sin_t[N_DFT];

void rp_init(void) {
    /* rotate (1, 0) by 2*pi/189 repeatedly; double precision keeps the error < 1e-13 */
    const double c1 = 0.9994474569326756, s1 = 0.03323824349762674;
    double c = 1.0, s = 0.0;
    for (int i = 0; i < N_DFT; i++) {
        cos_t[i] = (float)c;
        sin_t[i] = (float)s;
        double cn = c * c1 - s * s1;
        s = s * c1 + c * s1;
        c = cn;
    }
}

void rp_reset(rp_state_t *s) { s->n = 0; }

/* |analytic signal| of one range profile (length 189), only for the kept bins,
 * averaged in groups of RP_POOL. Matches scipy.signal.hilbert (N = 189, odd). */
static void envelope_pool(const float x[N_DFT], float out[RP_RBINS]) {
    float re[K_HALF], im[K_HALF];
    for (int k = 0; k < K_HALF; k++) {                 /* forward DFT, bins 0..94 */
        float sr = 0.f, si = 0.f;
        int idx = 0;
        for (int n = 0; n < N_DFT; n++) {
            sr += x[n] * cos_t[idx];
            si -= x[n] * sin_t[idx];
            idx += k; if (idx >= N_DFT) idx -= N_DFT;
        }
        float h = (k == 0) ? 1.f : 2.f;                /* Hilbert spectral weights */
        re[k] = h * sr; im[k] = h * si;
    }
    for (int r = 0; r < RP_RBINS; r++) {
        float acc = 0.f;
        for (int p = 0; p < RP_POOL; p++) {
            int m = RP_CROP0 + r * RP_POOL + p;        /* inverse DFT at kept bins only */
            float ar = 0.f, ai = 0.f;
            int idx = 0;
            for (int k = 0; k < K_HALF; k++) {
                ar += re[k] * cos_t[idx] - im[k] * sin_t[idx];
                ai += re[k] * sin_t[idx] + im[k] * cos_t[idx];
                idx += m; if (idx >= N_DFT) idx -= N_DFT;
            }
            acc += SQRTF(ar * ar + ai * ai) / (float)N_DFT;
        }
        out[r] = acc / (float)RP_POOL;
    }
}

int rp_push_frame(rp_state_t *s, const rp_frame_t frame) {
    float prof[N_DFT], pooled[RP_RBINS];
    if (s->n == 0)                                         /* filter starts at frame 0 */
        for (int b = 0; b < RP_BINS; b++)
            for (int c = 0; c < RP_CH; c++) s->bg[b][c] = frame[b][c];
    for (int c = 0; c < RP_CH; c++) {
        for (int b = 0; b < RP_BINS; b++) {                /* loop-back clutter filter */
            float v = frame[b][c];
            s->bg[b][c] = RP_ALPHA * s->bg[b][c] + (1.f - RP_ALPHA) * v;
            prof[b] = v - s->bg[b][c];
        }
        envelope_pool(prof, pooled);
        for (int r = 0; r < RP_RBINS; r++) s->win[s->n][r][c] = pooled[r];
    }
    s->n++;
    return s->n == RP_FRAMES;
}

/* ------------------------------------------------------------------ INT8 CNN */

static int8_t  in_q[RP_FRAMES * RP_RBINS * RP_CH];                 /* 90x32x3 */
static int8_t  act1[(RP_FRAMES / 2) * (RP_RBINS / 2) * C1_OUT];    /* 45x16x16 */
static int8_t  act2[(RP_FRAMES / 4) * (RP_RBINS / 4) * C2_OUT];    /* 22x8x32  */
static int32_t gap_acc[C3_OUT];

unsigned rp_inference_ram_bytes(void) {
    return (unsigned)(sizeof(in_q) + sizeof(act1) + sizeof(act2) + sizeof(gap_acc));
}

static inline int32_t round_half_up(float v) {       /* floor(v + 0.5) */
    float t = v + 0.5f;
    int32_t i = (int32_t)t;                             /* truncates toward zero */
    return ((float)i > t) ? i - 1 : i;
}

static inline int8_t conv_px(const int8_t *in, int H, int W, int C, int y, int x, int o,
                             const int8_t *w, const int32_t *b, const float *m, int zi, int zo) {
    int32_t acc = b[o];
    for (int ky = 0; ky < 3; ky++) {
        int iy = y + ky - 1;
        if (iy < 0 || iy >= H) continue;                   /* zero padding = input zero point */
        for (int kx = 0; kx < 3; kx++) {
            int ix = x + kx - 1;
            if (ix < 0 || ix >= W) continue;
            const int8_t *a = in + (iy * W + ix) * C;
            const int8_t *wk = w + ((o * 3 + ky) * 3 + kx) * C;
            for (int c = 0; c < C; c++) acc += (int32_t)(a[c] - zi) * wk[c];
        }
    }
    int32_t q = round_half_up((float)acc * m[o]) + zo;
    return (int8_t)(q < zo ? zo : (q > 127 ? 127 : q));    /* fused ReLU */
}

/* 3x3 conv + ReLU + 2x2 max-pool, fused (output H/2 x W/2, floor). */
static void conv_relu_pool(const int8_t *in, int H, int W, int C, int8_t *out, int O,
                           const int8_t *w, const int32_t *b, const float *m, int zi, int zo) {
    int H2 = H / 2, W2 = W / 2;
    for (int y2 = 0; y2 < H2; y2++)
        for (int x2 = 0; x2 < W2; x2++)
            for (int o = 0; o < O; o++) {
                int8_t v = -128;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        int8_t t = conv_px(in, H, W, C, 2 * y2 + dy, 2 * x2 + dx, o, w, b, m, zi, zo);
                        if (t > v) v = t;
                    }
                out[(y2 * W2 + x2) * O + o] = v;
            }
}

static int classify_quantized(int8_t scores[RP_CLASSES]) {
    conv_relu_pool(in_q, RP_FRAMES, RP_RBINS, C1_IN, act1, C1_OUT, C1_W, C1_B, C1_M, C1_IN_ZP, C1_OUT_ZP);
    conv_relu_pool(act1, RP_FRAMES / 2, RP_RBINS / 2, C2_IN, act2, C2_OUT, C2_W, C2_B, C2_M, C2_IN_ZP, C2_OUT_ZP);

    /* conv3 + ReLU fused with global average pooling */
    const int H = RP_FRAMES / 4, W = RP_RBINS / 4;          /* 22 x 8 */
    for (int o = 0; o < C3_OUT; o++) gap_acc[o] = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int o = 0; o < C3_OUT; o++)
                gap_acc[o] += conv_px(act2, H, W, C3_IN, y, x, o, C3_W, C3_B, C3_M, C3_IN_ZP, C3_OUT_ZP) - GAP_IN_ZP;
    int8_t g[C3_OUT];
    for (int o = 0; o < C3_OUT; o++) {
        int32_t q = round_half_up(((float)gap_acc[o] / (float)(H * W)) * GAP_M) + GAP_OUT_ZP;
        g[o] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }

    int best = 0;
    for (int o = 0; o < FC_OUT; o++) {                      /* classifier */
        int32_t acc = FC_B[o];
        for (int c = 0; c < FC_IN; c++) acc += (int32_t)(g[c] - FC_IN_ZP) * FC_W[o * FC_IN + c];
        int32_t q = round_half_up((float)acc * FC_M[o]) + FC_OUT_ZP;
        scores[o] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
        if (scores[o] > scores[best]) best = o;
    }
    return best;
}

int rp_classify_window(const float win[RP_FRAMES][RP_RBINS][RP_CH], int8_t scores[RP_CLASSES]) {
    const int n = RP_FRAMES * RP_RBINS * RP_CH;
    const float *x = &win[0][0][0];
    double sum = 0.0, sq = 0.0;                             /* per-recording scaling */
    for (int i = 0; i < n; i++) sum += x[i];
    double mean = sum / n;
    for (int i = 0; i < n; i++) { double d = x[i] - mean; sq += d * d; }
    float inv = 1.f / (SQRTF((float)(sq / n)) + 1e-8f);
    for (int i = 0; i < n; i++) {                           /* z-score + INT8 quantisation */
        int c = i % RP_CH;
        float z = (x[i] * inv - NORM_MEAN[c]) / NORM_STD[c];
        int32_t q = round_half_up(z / IN_SCALE) + IN_ZP;
        in_q[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
    return classify_quantized(scores);
}

int rp_classify(const rp_state_t *s, int8_t scores[RP_CLASSES]) {
    return rp_classify_window(s->win, scores);
}
