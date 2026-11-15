/* Host test: checks the C pipeline against Python/TFLite reference vectors. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "radar_pipeline.h"

#define WIN (RP_FRAMES * RP_RBINS * RP_CH)

static int test_inference(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { printf("cannot open %s\n", path); return 1; }
    int32_t n; fread(&n, 4, 1, f);
    float *X = malloc(sizeof(float) * WIN * n); int8_t *ref = malloc(12 * n); int32_t *y = malloc(4 * n);
    fread(X, sizeof(float), (size_t)WIN * n, f); fread(ref, 1, 12 * n, f); fread(y, 4, n, f); fclose(f);
    int same_argmax = 0, same_vec = 0, correct = 0, maxdiff = 0;
    for (int i = 0; i < n; i++) {
        int8_t s[RP_CLASSES];
        int p = rp_classify_window((const float (*)[RP_RBINS][RP_CH])(X + (size_t)i * WIN), s);
        int r = 0, vec = 1;
        for (int k = 0; k < 12; k++) {
            if (ref[i * 12 + k] > ref[i * 12 + r]) r = k;
            int d = abs(s[k] - ref[i * 12 + k]); if (d) vec = 0; if (d > maxdiff) maxdiff = d;
        }
        same_argmax += (p == r); same_vec += vec; correct += (p == y[i]);
    }
    printf("[inference] %d windows | argmax identical to TFLite: %d/%d | identical score vectors: %d/%d | max score diff: %d | accuracy: %.1f%%\n",
           n, same_argmax, n, same_vec, n, maxdiff, 100.0 * correct / n);
    free(X); free(ref); free(y);
    return same_argmax != n;
}

static int test_preprocessing(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { printf("cannot open %s\n", path); return 1; }
    int32_t n; fread(&n, 4, 1, f);
    size_t raw_sz = (size_t)RP_FRAMES * RP_BINS * RP_CH;
    float *raw = malloc(sizeof(float) * raw_sz * n), *cmp = malloc(sizeof(float) * WIN * n);
    fread(raw, sizeof(float), raw_sz * n, f); fread(cmp, sizeof(float), (size_t)WIN * n, f); fclose(f);
    static rp_state_t st; double max_rel = 0, max_abs = 0;
    for (int i = 0; i < n; i++) {
        rp_reset(&st);
        for (int t = 0; t < RP_FRAMES; t++)
            rp_push_frame(&st, (const float (*)[RP_CH])(raw + i * raw_sz + (size_t)t * RP_BINS * RP_CH));
        const float *w = &st.win[0][0][0], *e = cmp + (size_t)i * WIN;
        double peak = 0; for (int k = 0; k < WIN; k++) if (fabs(e[k]) > peak) peak = fabs(e[k]);
        for (int k = 0; k < WIN; k++) {
            double d = fabs(w[k] - e[k]); if (d > max_abs) max_abs = d;
            if (d / peak > max_rel) max_rel = d / peak;
        }
    }
    printf("[pre-processing] %d recordings | max |C - scipy| = %.2e (%.2e of peak)\n", n, max_abs, max_rel);
    free(raw); free(cmp);
    return max_rel > 1e-4;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : ".";
    char p1[512], p2[512];
    snprintf(p1, sizeof p1, "%s/infer_vectors.bin", dir); snprintf(p2, sizeof p2, "%s/prep_vectors.bin", dir);
    rp_init();
    printf("inference RAM (static buffers): %u bytes\n", rp_inference_ram_bytes());
    printf("state RAM (filter + window):    %u bytes\n", (unsigned)sizeof(rp_state_t));
    int fail = test_preprocessing(p2) | test_inference(p1);
    printf(fail ? "HOST TEST: FAIL\n" : "HOST TEST: PASS\n");
    return fail;
}
