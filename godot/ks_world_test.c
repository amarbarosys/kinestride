// G1 harness (docs/plans/GODOT_SPIKE.md): ks_create(DIR), STEPS x ks_step, ks_poses to OUT (float32 rows, [row][animal]),
// timing per call on stderr. The gate compares OUT with the engine binary's smooth.bin pose rows from the same DIR.
// PARTS=DIR/parts.txt also writes OUT.parts (ks_part of every part, [part][animal][12]) and OUT.qpos (ks_qpos, [NQ][F]).
// usage: cc godot/ks_world_test.c -ldl -o T && gpu-wait run -- T LIB.so DIR/ STEPS OUT
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
int main(int argc, char **argv) {
    if (argc < 5) { printf("FAIL usage: LIB DIR STEPS OUT\n"); return 2; }
    void *l = dlopen(argv[1], RTLD_NOW);
    if (!l) { printf("FAIL dlopen: %s\n", dlerror()); return 1; }
    long (*create)(const char *) = dlsym(l, "ks_create"); long (*step)(long) = dlsym(l, "ks_step");
    long (*poses)(long, float *) = dlsym(l, "ks_poses"); long (*npose)(void) = dlsym(l, "ks_npose"); void (*destroy)(long) = dlsym(l, "ks_destroy");
    long (*step_n)(long, long) = dlsym(l, "ks_step_n");
    long h = create(argv[2]);
    if (!h) { printf("FAIL ks_create: %s\n", argv[2]); return 1; }
    long n = npose(), steps = atol(argv[3]);
    float *p = malloc(n * sizeof(float));
    double t0 = now();
    if (getenv("STEP_N")) { if (step_n(h, steps)) { printf("FAIL ks_step_n\n"); return 1; } }
    else for (long s = 0; s < steps; s++) {
        /* PARTS: qpos taken before the last launch; with K=1 the pose rows are that state's kinematics */
        if (getenv("PARTS") && s == steps - 1) {
            long (*qpos)(long, float *) = dlsym(l, "ks_qpos"); float *q = malloc(n / 7 * 4 * sizeof(float));
            char path[4096]; snprintf(path, sizeof path, "%s.qpos", argv[4]);
            if (qpos(h, q)) { printf("FAIL ks_qpos\n"); return 1; }
            FILE *o = fopen(path, "wb"); fwrite(q, sizeof(float), atol(getenv("NQ")) * (n / 7 / 13), o); fclose(o);
        }
        if (step(h)) { printf("FAIL ks_step\n"); return 1; }
    }
    if (poses(h, p)) { printf("FAIL ks_poses\n"); return 1; }
    double t1 = now(); poses(h, p); double t2 = now();
    fprintf(stderr, "steps %ld: %.3f ms total (incl. first compile), poses copy %.3f ms for %ld floats\n", steps, t1 - t0, t2 - t1, n);
    FILE *f = fopen(argv[4], "wb"); fwrite(p, sizeof(float), n, f); fclose(f);
    if (getenv("PARTS")) {
        long (*fetch)(long) = dlsym(l, "ks_fetch"); void (*part)(long, long, float *, float *) = dlsym(l, "ks_part");
        FILE *pf = fopen(getenv("PARTS"), "r"); char path[4096];
        snprintf(path, sizeof path, "%s.parts", argv[4]); FILE *o = fopen(path, "wb");
        float hs[3], gq[4], *buf = malloc(n / 7 * 12 * sizeof(float));
        long parts = 0, F = 0;
        if (fetch(h)) { printf("FAIL ks_fetch\n"); return 1; }
        while (fscanf(pf, "%f %f %f %f %f %f %f", hs, hs + 1, hs + 2, gq, gq + 1, gq + 2, gq + 3) == 7) parts++;
        rewind(pf); F = n / 7 / parts;
        for (long p = 0; p < parts; p++) {
            if (fscanf(pf, "%f %f %f %f %f %f %f", hs, hs + 1, hs + 2, gq, gq + 1, gq + 2, gq + 3) != 7) return 1;
            part(h, p, gq, buf); fwrite(buf, sizeof(float), 12 * F, o);
        }
        fclose(o); fclose(pf);
    }
    destroy(h);
    printf("PASS ks_world %ld steps, %ld pose floats\n", steps, n);
    return 0;
}
