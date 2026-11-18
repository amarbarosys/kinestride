# G2 gate (private plan GODOT_SPIKE): KinestrideWorld and NpcMind from GDScript write what the C harnesses write, so
# spike_check.sh compares bytes. usage: godot --headless --path godot/spike --script g2_test.gd -- WORLD_DIR/ PARTS NPC_DIR/ OUT_DIR/
extends SceneTree

func _initialize() -> void:
	var a := OS.get_cmdline_user_args()
	var w := KinestrideWorld.new()
	if not w.open(a[0]) or not w.step(101) or not w.fetch():
		return fail("KinestrideWorld open/step/fetch")
	var parts := FileAccess.open(a[3] + "g2_parts.bin", FileAccess.WRITE)
	var p := 0
	for line in FileAccess.get_file_as_string(a[1]).split("\n", false):
		var v := line.split_floats(" ")
		parts.store_buffer(w.part(p, PackedFloat32Array([v[3], v[4], v[5], v[6]])).to_byte_array())
		p += 1
	parts.close()
	var n := NpcMind.new()
	if not n.open(a[2], 0):
		return fail("NpcMind open")
	var tr := FileAccess.open(a[3] + "g2_trace.bin", FileAccess.WRITE)
	for t in 200:
		n.tick(1)
		var s: PackedInt32Array = n.state()
		var xyf := PackedInt32Array()
		for i in 100:
			xyf.append_array(s.slice(4 * i, 4 * i + 3))
		tr.store_buffer(xyf.to_byte_array())
	tr.close()
	print("PASS g2 gdscript ", w.count(), " animals x ", p, " parts, 200 npc ticks")
	quit(0)

func fail(what: String) -> void:
	print("FAIL g2: ", what)
	quit(1)
