# Godot spike scene (private plan GODOT_SPIKE N3, G4). No user args: idle (the G0 runs).
#   -- crowd DIR/ PARTS [SHOT.png FRAMES]  Kinestride humanoids: one MultiMesh per part, one step + one pose fetch per frame
#   -- npc DIR/ SEED [SHOT.png FRAMES]     npc-mind NPCs on their 32 x 32 grid: NPCs, food, forges, wolves, trees
# Prints "G4 ..." timings (ms, mean over 60 frames) in crowd mode.
extends Node3D

const G := 32
var mode := ""
var world: KinestrideWorld
var mind: NpcMind
var mms: Array[MultiMesh] = []
var gq: Array[PackedFloat32Array] = []
var shot := ""
var frames := 0
var frame := 0
var acc := PackedFloat64Array([0, 0, 0, 0])
var tick_t := 0.0

func _ready() -> void:
	var a := OS.get_cmdline_user_args()
	if a.is_empty():
		return
	mode = a[0]
	if a.size() > 4:
		shot = a[3]
		frames = int(a[4])
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(25, -20, 0)  # z up: the light looks along -z, tilted
	sun.shadow_enabled = true
	add_child(sun)
	var env := WorldEnvironment.new()
	env.environment = Environment.new()
	env.environment.background_mode = Environment.BG_COLOR
	env.environment.background_color = Color(0.55, 0.65, 0.8)
	env.environment.ambient_light_color = Color(0.6, 0.6, 0.65)
	add_child(env)
	var cam := Camera3D.new()
	add_child(cam)
	if mode == "crowd":
		world = KinestrideWorld.new()
		if not world.open(a[1]):
			push_error("KinestrideWorld.open failed: " + a[1]); get_tree().quit(1); return
		var side := ceili(sqrt(world.count())) * 1.5
		floor_plane(Vector3(side / 2, side / 2, 0), side + 6, Color(0.35, 0.4, 0.3))
		for line in FileAccess.get_file_as_string(a[2]).split("\n", false):
			var v := line.split_floats(" ")
			var box := BoxMesh.new()
			box.size = Vector3(v[0], v[1], v[2]) * 2
			gq.append(PackedFloat32Array([v[3], v[4], v[5], v[6]]))
			mms.append(multimesh(box, world.count(), Color.from_hsv(gq.size() * 0.07, 0.5, 0.9)))
		cam.position = Vector3(side * 0.5, -side * 0.35, side * 0.45)
		cam.look_at(Vector3(side * 0.5, side * 0.45, 0), Vector3(0, 0, 1))
	elif mode == "npc":
		mind = NpcMind.new()
		if not mind.open(a[1], int(a[2])):
			push_error("NpcMind.open failed: " + a[1]); get_tree().quit(1); return
		floor_plane(Vector3(G / 2.0, G / 2.0, 0), G, Color(0.5, 0.6, 0.4))
		var cap := CapsuleMesh.new(); cap.radius = 0.3; cap.height = 1.0
		var ball := SphereMesh.new(); ball.radius = 0.25; ball.height = 0.5
		var cube := BoxMesh.new(); cube.size = Vector3(0.8, 0.8, 0.8)
		var trunk := CylinderMesh.new(); trunk.top_radius = 0.1; trunk.bottom_radius = 0.4; trunk.height = 1.6
		for spec in [[cap, 100, Color(0.9, 0.8, 0.6)], [ball, 60, Color(0.2, 0.8, 0.2)], [cube, 6, Color(0.5, 0.5, 0.5)],
				[cube, 3, Color(0.8, 0.1, 0.1)], [trunk, G * G, Color(0.1, 0.35, 0.15)]]:
			mms.append(multimesh(spec[0], spec[1], spec[2]))
		cam.position = Vector3(G / 2.0, -G * 0.45, G * 0.8)
		cam.look_at(Vector3(G / 2.0, G * 0.45, 0), Vector3(0, 0, 1))
		draw_npcs(mind.state())

func floor_plane(center: Vector3, size: float, c: Color) -> void:
	var m := MeshInstance3D.new()
	var p := PlaneMesh.new()
	p.size = Vector2(size, size)
	p.orientation = PlaneMesh.FACE_Z
	m.mesh = p
	m.position = center
	m.material_override = mat(c)
	add_child(m)

func mat(c: Color) -> StandardMaterial3D:
	var s := StandardMaterial3D.new()
	s.albedo_color = c
	return s

func multimesh(mesh: Mesh, n: int, c: Color) -> MultiMesh:
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.mesh = mesh
	mm.instance_count = n
	mesh.surface_set_material(0, mat(c))
	var inst := MultiMeshInstance3D.new()
	inst.multimesh = mm
	add_child(inst)
	return mm

# Grid cells to MultiMesh rows: identity basis rotated up (capsules and trees stand along z), at the cell centre.
func cells(mm: MultiMesh, s: PackedInt32Array, at: int, n: int, stride: int, z: float, show: Callable) -> void:
	var buf := PackedFloat32Array()
	buf.resize(12 * mm.instance_count)
	for i in n:
		var x := float(s[at + stride * i]) + 0.5
		var y := float(s[at + stride * i + 1]) + 0.5
		var on: bool = show.call(i)
		var k := 12 * i
		buf[k] = 1 if on else 0; buf[k + 6] = -1 if on else 0; buf[k + 9] = 1 if on else 0  # x->x, y->z, z->y: stand up
		buf[k + 3] = x; buf[k + 7] = y; buf[k + 11] = z
	mm.buffer = buf

func draw_npcs(s: PackedInt32Array) -> void:
	var nf := 4 * 100
	cells(mms[0], s, 0, 100, 4, 0.5, func(i): return true)
	cells(mms[1], s, nf, 60, 3, 0.25, func(i): return s[nf + 3 * i + 2] == 0)
	cells(mms[2], s, nf + 180, 6, 2, 0.4, func(i): return true)
	cells(mms[3], s, nf + 192, 3, 2, 0.4, func(i): return true)
	var trees := PackedInt32Array()  # tree cells as x, y pairs (row-major G x G map after the wolves)
	var t0 := nf + 198
	for c in G * G:
		trees.append_array([c % G, c / G] if s[t0 + c] == 1 else [0, 0])
	cells(mms[4], trees, 0, G * G, 2, 0.8, func(i): return s[t0 + i] == 1)

func _process(dt: float) -> void:
	if mode == "crowd":
		var t0 := Time.get_ticks_usec()
		world.step(1)
		var t1 := Time.get_ticks_usec()
		world.fetch()
		var t2 := Time.get_ticks_usec()
		for p in mms.size():
			mms[p].buffer = world.part(p, gq[p])
		var t3 := Time.get_ticks_usec()
		acc[0] += t1 - t0; acc[1] += t2 - t1; acc[2] += t3 - t2; acc[3] += dt * 1e6
		if frame % 60 == 59:
			print("G4 F=%d step %.2f fetch %.2f parts+upload %.2f frame %.2f ms" % [world.count(), acc[0] / 6e4, acc[1] / 6e4, acc[2] / 6e4, acc[3] / 6e4])
			acc = PackedFloat64Array([0, 0, 0, 0])
	elif mode == "npc":
		tick_t += dt
		if tick_t >= 0.2:
			tick_t = 0
			mind.tick(1)
			draw_npcs(mind.state())
	frame += 1
	if shot != "" and frame == frames:
		get_viewport().get_texture().get_image().save_png(shot)
		print("SHOT ", shot)
		get_tree().quit(0)
