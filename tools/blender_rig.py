# Blender side of tools/blender_check.sh: torso (free joint) + leg (knee hinge) as parented meshes and joint Empties,
# exported by the blender-mjcf add-on, then a floor and a knee servo added for the engine check.
# usage: blender -b --factory-startup -P tools/blender_rig.py -- PLUGIN.py OUTDIR
import bpy, sys, os, importlib.util
plugin, out = sys.argv[sys.argv.index('--') + 1:][:2]
spec = importlib.util.spec_from_file_location('mjcf_export', plugin)
mj = importlib.util.module_from_spec(spec); spec.loader.exec_module(mj); mj.register()
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()

def box(name, loc, size):
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.object; o.name = name; o.location = loc; o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # default location=True bakes pos into vertices
    return o

def empty(name, kind, loc):  # add-on: name contains "joint"; PLAIN_AXES free, CIRCLE hinge about the Empty's local y
    o = bpy.data.objects.new(name, None); o.empty_display_type = kind; o.location = loc
    bpy.context.collection.objects.link(o); return o

torso, leg = box('torso', (0, 0, 1), (0.4, 0.2, 0.2)), box('leg', (0.2, 0, 0.75), (0.08, 0.08, 0.5))
root, knee = empty('root_joint', 'PLAIN_AXES', (0, 0, 1)), empty('knee_joint', 'CIRCLE', (0.2, 0, 1.0))
bpy.context.view_layer.update()  # matrix_world is stale until the depsgraph runs
for c, p in ((leg, torso), (root, torso), (knee, leg)):
    w = c.matrix_world.copy(); c.parent = p; c.matrix_world = w
bpy.context.view_layer.update()
os.makedirs(out, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, 'rig.blend'))  # kinestride/blend.py input, same scene as rig.xml
assert bpy.ops.export_scene.mjcf_export(filepath=os.path.join(out, 'rig.xml'), export_format='OBJ') == {'FINISHED'}
s = open(os.path.join(out, 'rig.xml')).read()
s = s.replace('  <worldbody>\n', '  <worldbody>\n    <geom name="floor" type="plane" size="5 5 0.1"/>\n', 1)
s = s.replace('  <asset>', '  <actuator><position joint="knee_joint" kp="5"/></actuator>\n  <asset>', 1)
open(os.path.join(out, 'rig_floor.xml'), 'w').write(s)
