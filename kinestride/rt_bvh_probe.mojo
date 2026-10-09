# Compile probe: Mojo emits RDNA3 image_bvh_intersect_ray (gfx1100) via pop.call_llvm_intrinsic with a 3-wide vector.
# Run: .venv/bin/mojo run kinestride/rt_bvh_probe.mojo | grep image_bvh_intersect_ray
from std.sys import llvm_intrinsic
from max.gpu.host.compile import _compile_code
from max.gpu.host import get_gpu_target

comptime V3 = __mlir_type.`vector<3xf32>`

def k(res: UnsafePointer[Int32, MutAnyOrigin], node: Int32, x: Float32):
    var o = __mlir_op.`llvm.mlir.poison`[_type=V3]()
    var i0 = Int32(0)
    o = __mlir_op.`llvm.insertelement`[_type=V3](o, __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.f32](x._mlir_value), __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i32](i0._mlir_value))
    var desc = SIMD[DType.int32, 4](0, 0, 0, 0)
    var ext = Float32(1e30)
    var r = SIMD[DType.int32, 4](mlir_value=__mlir_op.`pop.call_llvm_intrinsic`[intrin = "llvm.amdgcn.image.bvh.intersect.ray.i32.v3f32".value, hasSideEffects = __mlir_attr.false, _type = SIMD[DType.int32, 4]._mlir_type](node._mlir_value, ext._mlir_value, o, o, o, desc._mlir_value))
    res[0] = r[0]

def main() raises:
    var a = _compile_code[k, target=get_gpu_target["amdgpu:gfx1100"]()]().asm
    print(a)
