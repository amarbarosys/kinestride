"""Standalone deterministic gate for the game state module and event replay."""

from __future__ import annotations

import subprocess
import struct
from pathlib import Path

from kinestride.game_events import Animal, Event, Rules, _self_check, replay


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "kinestride/game_state.mojo"


def main() -> None:
    mojo = Path(__file__).resolve().parents[1] / "tools/bin/mojo-cached"
    build_dir = ROOT / ".work/gamestate-check"
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "game_state"
    event_path = build_dir / "events.bin"
    mojo_rows_path = build_dir / "mojo-rows.bin"
    mojo_source_path = build_dir / "mojo-source.bin"
    subprocess.run([str(mojo), "build", str(MODULE), "-o", str(binary)],
                   cwd=ROOT, check=True)

    # One shared event stream: serialize it for Mojo and replay the same objects in Python.
    stream = [
        Event(0, resource_event="bite-a", source_id="berry", command_magnitude=999.0),
        Event(1, resource_event="bite-b", source_id="berry", command_magnitude=999.0),
        Event(0, work=1.0, resting=True, hazard_events=("trap-entry",), command_magnitude=0.0),
        Event(0, hazard_events=("trap-entry",), command_magnitude=123.0),
        Event(0, resource_event="bite-a", source_id="berry", command_magnitude=456.0),
        Event(1, resource_event="bite-b", source_id="berry", command_magnitude=789.0),
    ]
    hazard_ids: dict[str, int] = {}
    resource_ids: dict[str, int] = {}
    packed = bytearray()
    for event in stream:
        hazard_id = 0
        if event.hazard_events:
            hazard_id = hazard_ids.setdefault(event.hazard_events[0], len(hazard_ids) + 1)
        resource_id = 0
        if event.resource_event is not None:
            resource_id = resource_ids.setdefault(event.resource_event, len(resource_ids) + 1)
        source_id = 0 if event.source_id == "berry" else -1
        packed.extend(struct.pack("<ifiiiif", event.animal, event.work,
                                  int(event.resting), hazard_id, resource_id,
                                  source_id, event.command_magnitude))
    event_path.write_bytes(packed)
    subprocess.run([str(binary), str(event_path), str(mojo_rows_path), str(mojo_source_path)],
                   cwd=ROOT, check=True)

    initial = [Animal(), Animal()]
    sources = {"berry": 1}
    expected_rows = replay(initial, stream, sources, Rules(
        basal_drain=0.1, work_drain=0.1, hazard_damage=0.4,
        health_regen=0.0, rest_recovery=0.25, resource_energy=0.5))
    expected_bytes = b"".join(struct.pack("<4f", row.energy, row.health, row.stamina,
                                           float(row.dead)) for row in expected_rows)
    got_bytes = mojo_rows_path.read_bytes()
    assert got_bytes == expected_bytes, (
        f"Mojo/Python state-row byte mismatch: mojo={got_bytes.hex()} python={expected_bytes.hex()}"
    )
    got_source = struct.unpack("<i", mojo_source_path.read_bytes())[0]
    assert got_source == sources["berry"] == 0
    print("GAME_PARITY PASS: same 6-event stream, 2 state rows byte-identical; source_remaining=0")
    _self_check()


if __name__ == "__main__":
    main()
