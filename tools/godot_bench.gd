# Godot baseline for docs/provenance/motion-godot.md: TwoBoneIK3D solves/s (LEGS legs) and RetargetModifier3D
# retargets/s (HUMANS pairs of SkeletonProfileHumanoid skeletons, global pose) in normal use: Godot runs skeleton
# modifiers once per engine frame (deferred update, a ~6.9 ms floor per headless frame), so the bench times FRAMES
# engine frames over many skeletons (enough that work dominates the floor), then
# holds the targets still for one frame and checks every solve landed (a rate from unsolved skeletons is refused).
# usage: godot --headless --script tools/godot_bench.gd -- [FRAMES] [LEGS] [HUMANS]
extends SceneTree

var frames := 300
var legs := 1000
var humans := 100
var tick := 0
var t0 := 0
var leg_sk: Array[Skeleton3D] = []
var leg_target: Array[Node3D] = []
var leg_ik: Array[TwoBoneIK3D] = []
var miss := 0.0
var src: Array[Skeleton3D] = []
var dst: Array[Skeleton3D] = []


func _initialize() -> void:
	var a := OS.get_cmdline_user_args()
	if a.size() > 2:
		frames = int(a[0])
		legs = int(a[1])
		humans = int(a[2])
	for i in legs:
		build_leg()
	var profile := SkeletonProfileHumanoid.new()
	for i in humans:
		build_pair(profile)


func build_leg() -> void:
	var sk := Skeleton3D.new()
	root.add_child(sk)
	for i in 3:
		sk.add_bone(["hip", "knee", "ankle"][i])
		if i > 0:
			sk.set_bone_parent(i, i - 1)
			sk.set_bone_rest(i, Transform3D(Basis(), Vector3(0, -0.45, 0.04 if i == 1 else -0.04)))  # bent: a straight chain has no bend plane
	sk.reset_bone_poses()
	sk.modifier_callback_mode_process = Skeleton3D.MODIFIER_CALLBACK_MODE_PROCESS_IDLE
	var target := Node3D.new()
	root.add_child(target)
	var ik := TwoBoneIK3D.new()
	sk.add_child(ik)
	ik.set_setting_count(1)
	ik.set_root_bone(0, 0)
	ik.set_middle_bone(0, 1)
	ik.set_end_bone(0, 2)
	ik.set_target_node(0, ik.get_path_to(target))
	ik.set_pole_direction(0, SkeletonModifier3D.SECONDARY_DIRECTION_PLUS_Z)
	var pole := Node3D.new()  # without a pole node TwoBoneIK3D writes nothing
	root.add_child(pole)
	pole.position = Vector3(0, -0.45, 1)
	ik.set_pole_node(0, ik.get_path_to(pole))
	leg_sk.append(sk)
	leg_target.append(target)
	leg_ik.append(ik)


func humanoid(profile: SkeletonProfile) -> Skeleton3D:
	var sk := Skeleton3D.new()
	for i in profile.bone_size:
		sk.add_bone(profile.get_bone_name(i))
	for i in profile.bone_size:
		var p := profile.get_bone_parent(i)
		if p != &"":
			sk.set_bone_parent(i, sk.find_bone(p))
		sk.set_bone_rest(i, profile.get_reference_pose(i))
	sk.reset_bone_poses()
	sk.modifier_callback_mode_process = Skeleton3D.MODIFIER_CALLBACK_MODE_PROCESS_IDLE
	return sk


func build_pair(profile: SkeletonProfile) -> void:
	var s := humanoid(profile)
	root.add_child(s)
	var rt := RetargetModifier3D.new()
	rt.profile = profile
	rt.use_global_pose = true
	s.add_child(rt)
	var d := humanoid(profile)
	rt.add_child(d)
	src.append(s)
	dst.append(d)


func pose(f: int) -> void:
	for i in legs:
		leg_target[i].position = Vector3(0.1 * sin(f * 0.01 + i), -0.7 + 0.1 * cos(f * 0.013 + i), 0.05)
	for s in src:
		for b in s.get_bone_count():
			s.set_bone_pose_rotation(b, Quaternion(Vector3.UP, 0.3 * sin(f * 0.01 + b)))


func _process(_delta: float) -> bool:
	tick += 1
	if tick == 3:  # nodes are in the tree and ready
		t0 = Time.get_ticks_usec()
	if tick < 3 + frames:
		pose(tick)
		return false
	if tick == 3 + frames:  # targets held still; the next update's modifier output is measured inside its signal
		var dt := (Time.get_ticks_usec() - t0) / 1e6
		print("godot ", Engine.get_version_info().string, ": ", frames, " frames, ", legs, " legs, ", humans, " retarget pairs, ", dt, " s")
		print("all modifiers, per second: ", frames * legs / dt, " leg solves + ", frames * humans / dt, " retargets (one shared frame budget)")
		for i in legs:
			leg_ik[i].modification_processed.connect(func(): miss = max(miss, (leg_sk[i].get_bone_global_pose(2).origin - leg_target[i].position).length()))
		return false
	if tick == 4 + frames:
		return false
	var rot := 0.0
	var h := dst[0].find_bone(&"LeftHand") if humans > 0 else 0
	for i in humans:
		rot = max(rot, dst[i].get_bone_global_pose(h).basis.get_rotation_quaternion().angle_to(src[i].get_bone_global_pose(h).basis.get_rotation_quaternion()))
	print("check: worst ankle miss ", miss, " m (inside modification_processed), worst LeftHand rotation dst vs src ", rot, " rad")
	if miss > 1e-3 or rot > 1e-3:
		print("FAIL a modifier did not land: the rates above are not solve rates")
	return true
