/*
 * Privacy-safe radar gesture recognition: on-device pipeline.
 *
 * Streaming pre-processing (loop-back clutter filter, Hilbert envelope,
 * range crop + pooling) followed by the INT8 TinyRadarCNN. Pure C99, no
 * dynamic memory, no libraries beyond <math.h>; runs unchanged on a PC
 * (host test) and on a Cortex-M microcontroller.
 */
#ifndef RADAR_PIPELINE_H
#define RADAR_PIPELINE_H

#include <stdint.h>

#define RP_FRAMES      90   /* slow-time frames per gesture window   */
#define RP_BINS        189  /* range bins delivered by the radar     */
#define RP_CH          3    /* radars: left, top, right              */
#define RP_CROP0       34   /* first kept range bin                  */
#define RP_CROP1       162  /* one past the last kept range bin      */
#define RP_POOL        4    /* range pooling factor                  */
#define RP_RBINS       ((RP_CROP1 - RP_CROP0) / RP_POOL)   /* 32 */
#define RP_CLASSES     12
#define RP_ALPHA       0.9f /* loop-back filter coefficient          */

/* One radar frame as delivered by the sensors: [range bin][radar] */
typedef float rp_frame_t[RP_BINS][RP_CH];

typedef struct {
    float bg[RP_BINS][RP_CH];                     /* clutter background estimate      */
    float win[RP_FRAMES][RP_RBINS][RP_CH];        /* compact window being filled      */
    int   n;                                      /* frames received in this window   */
} rp_state_t;

/* Call once at start-up (builds the DFT twiddle table). */
void rp_init(void);

/* Start a new gesture window (resets the clutter filter, as in training). */
void rp_reset(rp_state_t *s);

/* Feed one radar frame. Returns 1 when the 90-frame window is complete. */
int rp_push_frame(rp_state_t *s, const rp_frame_t frame);

/* Classify a complete window. Writes the 12 INT8 output scores and returns the
 * predicted class index (0..11). */
int rp_classify(const rp_state_t *s, int8_t scores[RP_CLASSES]);

/* Classify an already pre-processed compact window (used by the host test). */
int rp_classify_window(const float win[RP_FRAMES][RP_RBINS][RP_CH], int8_t scores[RP_CLASSES]);

/* Bytes of static RAM used by the inference buffers (for reporting). */
unsigned rp_inference_ram_bytes(void);

#endif
