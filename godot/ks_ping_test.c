// G0 harness (docs/plans/GODOT_SPIKE.md): dlopen libkinestride.so from a non-Mojo host, call ks_ping, expect N(N-1)/2.
// usage: cc godot/ks_ping_test.c -o .work/godot/ks_ping_test && gpu-wait run -- .work/godot/ks_ping_test LIB.so
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char **argv) {
    void *h = dlopen(argc > 1 ? argv[1] : ".work/godot/libkinestride.so", RTLD_NOW);
    if (!h) { printf("FAIL dlopen: %s\n", dlerror()); return 1; }
    long (*ping)(void) = (long (*)(void))dlsym(h, "ks_ping");
    if (!ping) { printf("FAIL dlsym: ks_ping\n"); return 1; }
    long s = ping(), want = 1024L * 1023 / 2;
    printf("%s ks_ping %ld (want %ld)\n", s == want ? "PASS" : "FAIL", s, want);
    return s != want;
}
