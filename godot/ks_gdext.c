// GDExtension entry for Kinestride (docs/plans/GODOT_SPIKE.md). G0 stub: at scene init, call ks_ping from inside
// Godot's process (HIP next to Godot's renderer) and print the result; G2 grows this into class KinestrideWorld.
// build: cc -shared -fPIC -O2 -I .work/godot godot/ks_gdext.c -L.work/godot -lkinestride -Wl,-rpath,'$ORIGIN' -o .work/godot/libks_gdext.so
#include <stdio.h>
#include "gdextension_interface.h"
long ks_ping(void);

static void init(void *u, GDExtensionInitializationLevel l) {
    if (l != GDEXTENSION_INITIALIZATION_SCENE) return;
    long s = ks_ping(), want = 1024L * 1023 / 2;
    printf("%s ks_ping in godot %ld (want %ld)\n", s == want ? "PASS" : "FAIL", s, want);
    fflush(stdout);
}
static void deinit(void *u, GDExtensionInitializationLevel l) {}

GDExtensionBool ks_gdext_init(GDExtensionInterfaceGetProcAddress p, GDExtensionClassLibraryPtr lib, GDExtensionInitialization *r) {
    r->minimum_initialization_level = GDEXTENSION_INITIALIZATION_SCENE;
    r->userdata = 0; r->initialize = init; r->deinitialize = deinit;
    return 1;
}
