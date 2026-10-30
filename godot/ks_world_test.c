// G1 harness (docs/plans/GODOT_SPIKE.md): ks_create(DIR), STEPS x ks_step, ks_poses to OUT (float32 rows, [row][animal]),
// timing per call on stderr. The gate compares OUT with the engine binary's smooth.bin pose rows from the same DIR.
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
    else for (long s = 0; s < steps; s++) if (step(h)) { printf("FAIL ks_step\n"); return 1; }
    if (poses(h, p)) { printf("FAIL ks_poses\n"); return 1; }
    double t1 = now(); poses(h, p); double t2 = now();
    fprintf(stderr, "steps %ld: %.3f ms total (incl. first compile), poses copy %.3f ms for %ld floats\n", steps, t1 - t0, t2 - t1, n);
    FILE *f = fopen(argv[4], "wb"); fwrite(p, sizeof(float), n, f); fclose(f);
    destroy(h);
    printf("PASS ks_world %ld steps, %ld pose floats\n", steps, n);
    return 0;
}
