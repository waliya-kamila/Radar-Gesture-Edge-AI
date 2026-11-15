/*
 * Gesture-recognition firmware for a Cortex-M4.
 * Reads radar frames one at a time from the sensor stream, runs the streaming
 * pre-processing on every frame and the INT8 CNN after each 90-frame window,
 * and sends ONLY the recognised gesture over the serial port.
 */
#include <stdint.h>
#include "radar_pipeline.h"
#include "semihost.h"

static rp_state_t state;
static rp_frame_t frame;

int main(void) {
    rp_init();
    sh_puts("BOOT radar-gesture firmware, Cortex-M4F\r\n");
    sh_puts("INFO ram_inference_bytes="); sh_put_int(rp_inference_ram_bytes());
    sh_puts(" ram_state_bytes="); sh_put_int((long)sizeof(state)); sh_puts("\r\n");

    int fd = sh_open("stream.bin");
    if (fd < 0) { sh_puts("ERROR cannot open stream.bin\r\n"); return 1; }

    long frames = 0, windows = 0;
    rp_reset(&state);
    while (sh_read(fd, frame, (int)sizeof(frame)) == 0) {       /* one sensor frame */
        frames++;
        if (rp_push_frame(&state, (const float (*)[RP_CH])frame)) {
            int8_t scores[RP_CLASSES];
            int g = rp_classify(&state, scores);
            sh_puts("GESTURE window="); sh_put_int(windows);
            sh_puts(" frame="); sh_put_int(frames);
            sh_puts(" id=G"); sh_put_int(g + 1);
            sh_puts(" score="); sh_put_int(scores[g]); sh_puts("\r\n");
            windows++;
            rp_reset(&state);
        }
    }
    sh_puts("DONE frames="); sh_put_int(frames);
    sh_puts(" windows="); sh_put_int(windows); sh_puts("\r\n");
    return 0;
}
