# Blender side of kinestride/blend.py (runs in Blender's bpy Python): exports a .blend with the blender-mjcf add-on.
# usage: bl run tools/blend_export.py -- FILE.blend PLUGIN.py OUT.xml
import bpy, sys, importlib.util
blend, plugin, out = sys.argv[sys.argv.index('--') + 1:][:3]
bpy.ops.wm.open_mainfile(filepath=blend)
spec = importlib.util.spec_from_file_location('mjcf_export', plugin)
mj = importlib.util.module_from_spec(spec); spec.loader.exec_module(mj); mj.register()
assert bpy.ops.export_scene.mjcf_export(filepath=out, export_format='OBJ') == {'FINISHED'}
