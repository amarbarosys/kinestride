// GDExtension entry for Kinestride (private plan GODOT_SPIKE). G0: at scene init, ks_ping from inside Godot's process
// (prints PASS/FAIL). G2: classes KinestrideWorld (libkinestride.so) and NpcMind (libks_npc.so), RefCounted, one method
// table and one dispatcher behind both Godot call paths (Variant call and ptrcall).
// build: cc -shared -fPIC -O2 -I .work/godot godot/ks_gdext.c -L.work/godot -lkinestride -lks_npc -Wl,-rpath,'$ORIGIN' -o .work/godot/libks_gdext.so
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gdextension_interface.h"
long ks_ping(void), ks_create(const char *), ks_step_n(long, long), ks_fetch(long), ks_nf(void);
void ks_part(long, long, const float *, float *), ks_destroy(long);
long ks_npc_create(const char *, long), ks_npc_tick(long, long), ks_npc_nstate(void), ks_npc_think(long, const float *, int32_t *);
void ks_npc_state(long, int32_t *), ks_npc_destroy(long);

#define T(x) GDEXTENSION_VARIANT_TYPE_##x
typedef struct { uint8_t b[8]; } SName;
typedef struct { uint8_t b[16]; } Packed;
static GDExtensionClassLibraryPtr lib;
static GDExtensionInterfaceStringNameNewWithLatin1Chars sname;
static GDExtensionInterfaceStringNewWithUtf8Chars snew;
static GDExtensionInterfaceStringToUtf8Chars sutf8;
static GDExtensionInterfaceClassdbConstructObject2 construct;
static GDExtensionInterfaceObjectSetInstance set_instance;
static GDExtensionInterfaceGetVariantFromTypeConstructor from_type;
static GDExtensionInterfaceGetVariantToTypeConstructor to_type;
static GDExtensionInterfaceVariantGetPtrConstructor ctor;
static GDExtensionInterfaceVariantGetPtrDestructor dtor;
static GDExtensionInterfaceVariantGetPtrBuiltinMethod builtin;
static GDExtensionInterfacePackedFloat32ArrayOperatorIndex f32_at;
static GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst f32_at_c;
static GDExtensionInterfacePackedInt32ArrayOperatorIndex i32_at;
static SName sn_world, sn_npc, sn_ref, sn_empty;
static uint8_t s_empty[8];

typedef struct { long h; int npc; } Inst;
enum { W_OPEN, W_STEP, W_FETCH, W_PART, W_COUNT, N_OPEN, N_TICK, N_STATE, N_THINK, NMETHOD };
// per method: name, class (1 = NpcMind), return type, argument types (NIL ends)
static const struct { const char *name; int npc; GDExtensionVariantType ret, arg[2]; } M[NMETHOD] = {
    {"open", 0, T(INT), {T(STRING), T(NIL)}},    {"step", 0, T(INT), {T(INT), T(NIL)}},
    {"fetch", 0, T(INT), {T(NIL)}},              {"part", 0, T(PACKED_FLOAT32_ARRAY), {T(INT), T(PACKED_FLOAT32_ARRAY)}},
    {"count", 0, T(INT), {T(NIL)}},              {"open", 1, T(INT), {T(STRING), T(INT)}},
    {"tick", 1, T(INT), {T(INT), T(NIL)}},       {"state", 1, T(PACKED_INT32_ARRAY), {T(NIL)}},
    {"think", 1, T(PACKED_INT32_ARRAY), {T(PACKED_FLOAT32_ARRAY), T(NIL)}},
};

// Resize a constructed packed array to n and return its data pointer.
static void *sized(Packed *a, GDExtensionVariantType t, int64_t n) {
    static GDExtensionPtrBuiltInMethod rf, ri;
    if (!rf) { SName r; sname(&r, "resize", 1); rf = builtin(T(PACKED_FLOAT32_ARRAY), &r, 848867239); ri = builtin(T(PACKED_INT32_ARRAY), &r, 848867239); }
    int64_t ok; const void *args[] = {&n};
    (t == T(PACKED_FLOAT32_ARRAY) ? rf : ri)(a, args, &ok, 1);
    return n == 0 ? NULL : t == T(PACKED_FLOAT32_ARRAY) ? (void *)f32_at(a, 0) : (void *)i32_at(a, 0);
}

// The one dispatcher: typed argument pointers in (int64_t, String, packed array), result into an int64_t or a constructed array.
static void run(int id, Inst *s, const void **a, void *ret) {
    char path[4096];
    int64_t r = 0;
    if (M[id].arg[0] == T(STRING)) sutf8(a[0], path, sizeof path - 1), path[sutf8(a[0], NULL, 0)] = 0;
    switch (id) {
    case W_OPEN: if (s->h) ks_destroy(s->h); s->h = ks_create(path); r = s->h != 0; break;
    case W_STEP: r = s->h ? ks_step_n(s->h, *(const int64_t *)a[0]) == 0 : 0; break;
    case W_FETCH: r = s->h ? ks_fetch(s->h) == 0 : 0; break;
    case W_COUNT: r = ks_nf(); break;
    case W_PART: {
        float *dst = sized(ret, T(PACKED_FLOAT32_ARRAY), 12 * ks_nf());
        if (s->h) ks_part(s->h, *(const int64_t *)a[0], f32_at_c(a[1], 0), dst);
        return;
    }
    case N_OPEN: if (s->h) ks_npc_destroy(s->h); s->h = ks_npc_create(path, *(const int64_t *)a[1]); r = s->h != 0; break;
    case N_TICK: r = s->h ? ks_npc_tick(s->h, *(const int64_t *)a[0]) == 0 : 0; break;
    case N_STATE: { int32_t *dst = sized(ret, T(PACKED_INT32_ARRAY), ks_npc_nstate()); if (s->h) ks_npc_state(s->h, dst); return; }
    case N_THINK: { int32_t *dst = sized(ret, T(PACKED_INT32_ARRAY), 300); if (s->h) ks_npc_think(s->h, f32_at_c(a[0], 0), dst); return; }
    }
    *(int64_t *)ret = r;
}

static void ptrcall(void *ud, GDExtensionClassInstancePtr inst, const GDExtensionConstTypePtr *args, GDExtensionTypePtr ret) {
    run((int)(intptr_t)ud, inst, (const void **)args, ret);
}

// Variant path: unpack each argument to its type, run, pack the result, destroy the temporaries.
static void call(void *ud, GDExtensionClassInstancePtr inst, const GDExtensionConstVariantPtr *args, GDExtensionInt argc,
                 GDExtensionVariantPtr ret, GDExtensionCallError *err) {
    int id = (int)(intptr_t)ud, n = 0;
    while (n < 2 && M[id].arg[n] != T(NIL)) n++;
    if (argc != n) { err->error = GDEXTENSION_CALL_ERROR_INVALID_ARGUMENT; err->expected = n; return; }
    uint8_t buf[2][16]; const void *a[2];
    for (int i = 0; i < n; i++) to_type(M[id].arg[i])(buf[i], (GDExtensionVariantPtr)args[i]), a[i] = buf[i];
    uint8_t out[16];
    if (M[id].ret != T(INT)) ctor(M[id].ret, 0)(out, NULL);
    run(id, inst, a, out);
    from_type(M[id].ret)(ret, out);
    if (M[id].ret != T(INT)) dtor(M[id].ret)(out);
    for (int i = 0; i < n; i++) if (M[id].arg[i] != T(INT)) dtor(M[id].arg[i])(buf[i]);
    err->error = GDEXTENSION_CALL_OK;
}

static GDExtensionObjectPtr create(void *npc, GDExtensionBool post) {
    GDExtensionObjectPtr o = construct(&sn_ref);
    Inst *s = calloc(1, sizeof *s); s->npc = npc != NULL;
    set_instance(o, npc ? &sn_npc : &sn_world, s);
    return o;
}
static void destroy(void *ud, GDExtensionClassInstancePtr p) {
    Inst *s = p;
    if (s->h) (s->npc ? ks_npc_destroy : ks_destroy)(s->h);
    free(s);
}

static void init(void *u, GDExtensionInitializationLevel l) {
    if (l != GDEXTENSION_INITIALIZATION_SCENE) return;
    long s = ks_ping(), want = 1024L * 1023 / 2;
    printf("%s ks_ping in godot %ld (want %ld)\n", s == want ? "PASS" : "FAIL", s, want);
    fflush(stdout);
    GDExtensionInterfaceClassdbRegisterExtensionClass4 reg_class = 0;
    GDExtensionInterfaceClassdbRegisterExtensionClassMethod reg_method = 0;
    extern GDExtensionInterfaceGetProcAddress gpa;
    reg_class = (void *)gpa("classdb_register_extension_class4");
    reg_method = (void *)gpa("classdb_register_extension_class_method");
    sname(&sn_world, "KinestrideWorld", 1); sname(&sn_npc, "NpcMind", 1); sname(&sn_ref, "RefCounted", 1); sname(&sn_empty, "", 1);
    snew(s_empty, "");
    for (int c = 0; c < 2; c++) {
        GDExtensionClassCreationInfo4 info = {0};
        info.is_exposed = 1; info.create_instance_func = create; info.free_instance_func = destroy;
        info.class_userdata = c ? (void *)1 : NULL;
        reg_class(lib, c ? &sn_npc : &sn_world, &sn_ref, &info);
    }
    static SName names[NMETHOD], argn[2];
    sname(&argn[0], "a", 1); sname(&argn[1], "b", 1);
    for (int id = 0; id < NMETHOD; id++) {
        GDExtensionPropertyInfo retp = {M[id].ret, &sn_empty, &sn_empty, 0, s_empty, 6}, argp[2];
        GDExtensionClassMethodArgumentMetadata meta[2] = {0, 0};
        int n = 0;
        while (n < 2 && M[id].arg[n] != T(NIL)) { argp[n] = (GDExtensionPropertyInfo){M[id].arg[n], &argn[n], &sn_empty, 0, s_empty, 6}; n++; }
        sname(&names[id], M[id].name, 1);
        GDExtensionClassMethodInfo mi = {&names[id], (void *)(intptr_t)id, call, ptrcall, GDEXTENSION_METHOD_FLAG_NORMAL,
                                         1, &retp, GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE, n, argp, meta, 0, NULL};
        reg_method(lib, M[id].npc ? &sn_npc : &sn_world, &mi);
    }
}
static void deinit(void *u, GDExtensionInitializationLevel l) {}

GDExtensionInterfaceGetProcAddress gpa;
GDExtensionBool ks_gdext_init(GDExtensionInterfaceGetProcAddress p, GDExtensionClassLibraryPtr l, GDExtensionInitialization *r) {
    gpa = p; lib = l;
    sname = (void *)p("string_name_new_with_latin1_chars"); snew = (void *)p("string_new_with_utf8_chars");
    sutf8 = (void *)p("string_to_utf8_chars"); construct = (void *)p("classdb_construct_object2");
    set_instance = (void *)p("object_set_instance"); from_type = (void *)p("get_variant_from_type_constructor");
    to_type = (void *)p("get_variant_to_type_constructor"); ctor = (void *)p("variant_get_ptr_constructor");
    dtor = (void *)p("variant_get_ptr_destructor"); builtin = (void *)p("variant_get_ptr_builtin_method");
    f32_at = (void *)p("packed_float32_array_operator_index"); f32_at_c = (void *)p("packed_float32_array_operator_index_const");
    i32_at = (void *)p("packed_int32_array_operator_index");
    r->minimum_initialization_level = GDEXTENSION_INITIALIZATION_SCENE;
    r->userdata = 0; r->initialize = init; r->deinitialize = deinit;
    return 1;
}
