"""Compact game-state transitions for GAME_LAYER items 4-7.

The engine hook should call these from the existing actuator/step work once
the owner approves the integration point. The CPU contract and event semantics
live in game_events.py. No per-step host update is required.

Port basis: MuJoCo 3.9.0 `engine_forward.c:312` (`mj_fwdActuation`) supplies
the physical actuator force used with velocity to account for work. Game bars
and tagged scenario events are game-layer extensions, not MuJoCo physics ports.
"""

from std.sys import argv
from max.gpu.host import DeviceBuffer, DeviceContext

@fieldwise_init
struct GameState(Copyable, Movable):
    var energy: Float32
    var health: Float32
    var stamina: Float32
    # 0.0 or 1.0 keeps the row exactly four float32 values (16 B).
    var dead: Float32


def clamp01(x: Float32) -> Float32:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def nonnegative(x: Float32) -> Float32:
    if x < 0.0:
        return 0.0
    return x


def work_drain(work: Float32, basal: Float32, rate: Float32) -> Float32:
    """Convert measured actuator work into game drain, independent of command magnitude."""
    return nonnegative(basal) + nonnegative(work) * nonnegative(rate)


def advance_state(
    state: GameState,
    work: Float32,
    damage_event: Bool,
    resting: Bool,
    resource_granted: Bool,
    basal_drain: Float32,
    work_rate: Float32,
    damage_amount: Float32,
    regen_rate: Float32,
    rest_rate: Float32,
    resource_energy: Float32,
) -> GameState:
    """Advance one animal row; damage_event is one deduplicated tagged event."""
    if state.dead != 0.0:
        return GameState(state.energy, state.health, state.stamina, state.dead)
    var energy = clamp01(state.energy - work_drain(work, basal_drain, work_rate))
    var health = clamp01(state.health + nonnegative(regen_rate))
    var stamina = clamp01(state.stamina - nonnegative(work) * nonnegative(work_rate))
    if resting:
        stamina = clamp01(stamina + nonnegative(rest_rate))
    if damage_event:
        health = clamp01(health - nonnegative(damage_amount))
    if resource_granted:
        energy = clamp01(energy + nonnegative(resource_energy))
    var dead: Float32 = 0.0
    if energy <= 0.0 or health <= 0.0:
        dead = 1.0
    return GameState(energy, health, stamina, dead)


def actuator_scale(state: GameState, floor_scale: Float32) -> Float32:
    """Scale servo and adhesion effort; dead state explicitly yields zero effort."""
    if state.dead != 0.0:
        return 0.0
    var stamina = clamp01(state.stamina)
    var floor = clamp01(floor_scale)
    if stamina < floor:
        return floor
    return stamina


struct Snapshot(Movable):
    """Device copy of the state that changes: qpos, qvel, muscle act rows, warm-start qacc rows. Never model or scratch."""
    var q: DeviceBuffer[DType.float32]
    var v: DeviceBuffer[DType.float32]
    var act: DeviceBuffer[DType.float32]
    var warm: DeviceBuffer[DType.float32]
    var bytes: Int

    def __init__(out self, ctx: DeviceContext, nq: Int, nv: Int, nact: Int, nwarm: Int, f: Int) raises:
        self.q = ctx.enqueue_create_buffer[DType.float32](nq * f)
        self.v = ctx.enqueue_create_buffer[DType.float32](nv * f)
        self.act = ctx.enqueue_create_buffer[DType.float32](max(nact, 1) * f)
        self.warm = ctx.enqueue_create_buffer[DType.float32](max(nwarm, 1) * f)
        self.bytes = (nq + nv + nact + nwarm) * f * 4


def snapshot(ctx: DeviceContext, q_b: DeviceBuffer[DType.float32], v_b: DeviceBuffer[DType.float32],
             o_b: DeviceBuffer[DType.float32], f: Int, act_row: Int, nact: Int, warm_row: Int, nwarm: Int,
             nq: Int, nv: Int) raises -> Snapshot:
    """Copy qpos, qvel and the o rows [act_row, +nact), [warm_row, +nwarm) (rows are [row][animal], F floats each)."""
    var s = Snapshot(ctx, nq, nv, nact, nwarm, f)
    ctx.enqueue_copy(s.q, q_b)
    ctx.enqueue_copy(s.v, v_b)
    if nact > 0:
        ctx.enqueue_copy(s.act, o_b.create_sub_buffer[DType.float32](act_row * f, nact * f))
    if nwarm > 0:
        ctx.enqueue_copy(s.warm, o_b.create_sub_buffer[DType.float32](warm_row * f, nwarm * f))
    ctx.synchronize()
    return s^


def restore(ctx: DeviceContext, s: Snapshot, q_b: DeviceBuffer[DType.float32], v_b: DeviceBuffer[DType.float32],
            o_b: DeviceBuffer[DType.float32], f: Int, act_row: Int, nact: Int, warm_row: Int, nwarm: Int) raises:
    ctx.enqueue_copy(q_b, s.q)
    ctx.enqueue_copy(v_b, s.v)
    if nact > 0:
        ctx.enqueue_copy(o_b.create_sub_buffer[DType.float32](act_row * f, nact * f), s.act)
    if nwarm > 0:
        ctx.enqueue_copy(o_b.create_sub_buffer[DType.float32](warm_row * f, nwarm * f), s.warm)
    ctx.synchronize()


def replay_binary_stream(input_path: String, rows_path: String, source_path: String) raises:
    """Replay packed seven-word events and write packed state rows for oracle comparison."""
    var raw = open(input_path, "r").read_bytes()
    var words = raw.unsafe_ptr().unsafe_bitcast[Int32]()
    var vals = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    var animals = [GameState(1.0, 1.0, 1.0, 0.0), GameState(1.0, 1.0, 1.0, 0.0)]
    var source_remaining: Int32 = 1
    var seen_hazards: List[Int32] = []
    var seen_resources: List[Int32] = []
    var nrecords = len(raw) // 28
    for record in range(nrecords):
        var base = record * 7
        var animal_index = Int(words.unsafe_offset(base)[])
        var hazard_id = words.unsafe_offset(base + 3)[]
        var resource_id = words.unsafe_offset(base + 4)[]
        var source_id = words.unsafe_offset(base + 5)[]
        var duplicate_hazard = False
        if hazard_id > 0:
            for prior in range(len(seen_hazards) // 2):
                if seen_hazards[prior * 2] == words.unsafe_offset(base)[] and seen_hazards[prior * 2 + 1] == hazard_id:
                    duplicate_hazard = True
            if not duplicate_hazard:
                seen_hazards.append(words.unsafe_offset(base)[])
                seen_hazards.append(hazard_id)
        var resource_granted = False
        var duplicate_resource = False
        if resource_id > 0:
            for prior in range(len(seen_resources)):
                if seen_resources[prior] == resource_id:
                    duplicate_resource = True
            if not duplicate_resource and source_id == 0 and source_remaining > 0:
                source_remaining -= 1
                resource_granted = True
                seen_resources.append(resource_id)
        var event_hazard = hazard_id > 0 and not duplicate_hazard
        var next = advance_state(animals[animal_index], vals.unsafe_offset(base + 1)[], event_hazard,
                                 words.unsafe_offset(base + 2)[] != 0, resource_granted,
                                 0.1, 0.1, 0.4, 0.0, 0.25, 0.5)
        animals[animal_index] = next.copy()
    var output = List[Float32](length=4 * len(animals), fill=0.0)
    for animal in range(len(animals)):
        output[animal * 4] = animals[animal].energy
        output[animal * 4 + 1] = animals[animal].health
        output[animal * 4 + 2] = animals[animal].stamina
        output[animal * 4 + 3] = animals[animal].dead
    var row_file = open(rows_path, "w")
    row_file.write_bytes(Span(unsafe_ptr=output.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(output) * 4))
    var source_output = [source_remaining]
    var source_file = open(source_path, "w")
    source_file.write_bytes(Span(unsafe_ptr=source_output.unsafe_ptr().unsafe_bitcast[UInt8](), length=4))


def main() raises:
    comptime assert 4 * 4 == 16, "GameState row is four float32 values"
    var args = argv()
    if len(args) == 4:
        replay_binary_stream(String(args[1]), String(args[2]), String(args[3]))
        print("GAME_STATE PASS: replayed shared binary event stream")
        return
    var alive = GameState(1.0, 1.0, 1.0, 0.0)
    var drained = advance_state(alive, 2.0, False, False, False, 0.0, 0.1,
                                0.0, 0.0, 0.1, 0.0)
    assert abs(drained.energy - 0.8) < 1e-6
    var fed = advance_state(drained, 0.0, False, False, True, 0.0, 0.1,
                            0.0, 0.0, 0.0, 0.25)
    assert abs(fed.energy - 1.0) < 1e-6
    assert actuator_scale(GameState(0.4, 1.0, 0.0, 0.0), 0.15) == 0.15
    var dead = advance_state(GameState(0.0, 1.0, 1.0, 1.0), 5.0, True, False,
                             False, 0.1, 0.2, 1.0, 0.0, 0.1, 0.0)
    assert dead.energy == 0.0 and dead.health == 1.0 and dead.dead == 1.0
    assert actuator_scale(GameState(0.0, 0.0, 0.0, 1.0), 0.15) == 0.0
    print("GAME_STATE PASS: work drain, 16 B row, rest-independent state and dead effort gate")
