// N1 harness (docs/plans/GODOT_SPIKE.md): ks_npc_create(DIR, SEED), TICKS x (ks_npc_tick, ks_npc_state), writes the
// per-tick NPC x, y, face (int32 [N][3]) to OUT; the gate compares it with npc_live's `base grid` trace.
// usage: cc godot/ks_npc_test.c -ldl -o T && gpu-wait run -- T LIB.so DIR/ SEED TICKS OUT
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#define N 100
int main(int argc, char **argv) {
    if (argc < 6) { printf("FAIL usage: LIB DIR SEED TICKS OUT\n"); return 2; }
    void *l = dlopen(argv[1], RTLD_NOW);
    if (!l) { printf("FAIL dlopen: %s\n", dlerror()); return 1; }
    long (*create)(const char *, long) = dlsym(l, "ks_npc_create"); long (*tick)(long, long) = dlsym(l, "ks_npc_tick");
    void (*state)(long, int *) = dlsym(l, "ks_npc_state"); long (*nstate)(void) = dlsym(l, "ks_npc_nstate");
    void (*destroy)(long) = dlsym(l, "ks_npc_destroy");
    long h = create(argv[2], atol(argv[3])), ticks = atol(argv[4]);
    if (!h) { printf("FAIL ks_npc_create\n"); return 1; }
    int *s = malloc(nstate() * sizeof(int)), xyf[3 * N];
    FILE *f = fopen(argv[5], "wb");
    for (long t = 0; t < ticks; t++) {
        if (tick(h, 1)) { printf("FAIL ks_npc_tick\n"); return 1; }
        state(h, s);
        for (int i = 0; i < N; i++) for (int c = 0; c < 3; c++) xyf[3 * i + c] = s[4 * i + c];
        fwrite(xyf, sizeof(int), 3 * N, f);
    }
    fclose(f); destroy(h);
    printf("PASS ks_npc %ld ticks\n", ticks);
    return 0;
}
