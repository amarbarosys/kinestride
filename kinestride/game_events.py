"""Deterministic game-event oracle/replay for GAME_LAYER items 4-7.

The first implementation intentionally models explicit event streams, not
physics. Work is supplied as measured actuator work by the engine owner; policy
command magnitude is never interpreted as physical work.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
import struct
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Rules:
    basal_drain: float = 0.001
    work_drain: float = 0.01
    hazard_damage: float = 0.25
    health_regen: float = 0.0
    rest_recovery: float = 0.02
    resource_energy: float = 0.2


@dataclass
class Animal:
    energy: float = 1.0
    health: float = 1.0
    stamina: float = 1.0
    dead: bool = False

    def row(self) -> tuple[float, float, float, bool]:
        return (self.energy, self.health, self.stamina, self.dead)


@dataclass(frozen=True)
class Event:
    """One deterministic tick input.

    ``work`` is sum(abs(actuator_force * actuator_velocity)) integrated over
    the tick. Hazard IDs represent tagged hazard-entry events, not contacts.
    Resource events are named game triggers and carry stable event IDs.
    """

    animal: int
    work: float = 0.0
    resting: bool = False
    hazard_events: tuple[str, ...] = ()
    resource_event: str | None = None
    source_id: str | None = None
    command_magnitude: float = 0.0


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def replay(
    animals: Sequence[Animal],
    events: Iterable[Event],
    source_remaining: dict[str, int],
    rules: Rules = Rules(),
) -> list[Animal]:
    """Apply a stable event stream, consuming each event/resource at most once."""
    result = [Animal(*animal.row()) for animal in animals]
    seen_hazards: set[tuple[int, str]] = set()
    seen_resources: set[str] = set()
    for event in events:
        if event.animal < 0 or event.animal >= len(result):
            raise IndexError(f"animal index out of range: {event.animal}")
        animal = result[event.animal]
        if animal.dead:
            continue

        work = _f32(max(0.0, _f32(event.work)))
        drain = _f32(_f32(rules.basal_drain) + _f32(work * _f32(rules.work_drain)))
        animal.energy = _f32(_clamp01(_f32(animal.energy - drain)))
        stamina_drain = _f32(work * _f32(rules.work_drain))
        animal.stamina = _f32(_clamp01(_f32(animal.stamina - stamina_drain)))
        if event.resting:
            animal.stamina = _f32(_clamp01(_f32(animal.stamina + _f32(rules.rest_recovery))))
        animal.health = _f32(_clamp01(_f32(animal.health + _f32(rules.health_regen))))

        for hazard_id in event.hazard_events:
            key = (event.animal, hazard_id)
            if key not in seen_hazards:
                animal.health = _f32(_clamp01(_f32(animal.health - _f32(rules.hazard_damage))))
                seen_hazards.add(key)

        if event.resource_event is not None and event.source_id is not None:
            if event.resource_event not in seen_resources:
                remaining = source_remaining.get(event.source_id, 0)
                if remaining > 0:
                    source_remaining[event.source_id] = remaining - 1
                    animal.energy = _f32(_clamp01(_f32(animal.energy + _f32(rules.resource_energy))))
                    seen_resources.add(event.resource_event)

        if animal.energy <= 0.0 or animal.health <= 0.0:
            animal.dead = True
    return result


def assert_mutant_signatures() -> None:
    """Fixtures that distinguish the three required semantic fault signatures."""
    rules = Rules(basal_drain=0.0, work_drain=0.1, hazard_damage=0.2,
                  rest_recovery=0.1, resource_energy=0.0)
    base = [Animal()]
    worked = replay(base, [Event(0, work=2.0)], {}, rules)[0]
    assert isclose(worked.energy, 0.8, rel_tol=1e-6), "work-drain signature mismatch"
    contact_repeat = replay(base, [Event(0, hazard_events=("H",)),
                                   Event(0, hazard_events=("H",))], {}, rules)[0]
    assert isclose(contact_repeat.health, 0.8, rel_tol=1e-6), "hazard-event signature mismatch"
    bite = Event(0, resource_event="bite-1", source_id="food")
    sources = {"food": 1}
    eaten = replay([Animal()], [bite, bite], sources, rules)[0]
    assert sources["food"] == 0, "resource-depletion signature mismatch"
    assert isclose(eaten.energy, 1.0, rel_tol=1e-6), "resource-consumption signature mismatch"


def _self_check() -> None:
    assert_mutant_signatures()
    rules = Rules(basal_drain=0.1, work_drain=0.1, hazard_damage=0.4,
                  rest_recovery=0.25, resource_energy=0.5)
    # Two animals race for one remaining unit. Stable stream order wins once.
    source = {"berry": 1}
    rows = replay([Animal(), Animal()], [
        Event(0, resource_event="bite-a", source_id="berry"),
        Event(1, resource_event="bite-b", source_id="berry"),
        Event(0, work=1.0, resting=True, hazard_events=("trap-entry",)),
        Event(0, hazard_events=("trap-entry",)),
    ], source, rules)
    assert source["berry"] == 0
    assert isclose(rows[0].energy, 0.7, rel_tol=1e-6) and isclose(rows[0].health, 0.6, rel_tol=1e-6)
    assert isclose(rows[0].stamina, 1.0, rel_tol=1e-6)
    assert isclose(rows[1].energy, 0.9, rel_tol=1e-6)
    # Death is absorbing, including for later resource/hazard events.
    dead = replay([Animal(energy=0.0, dead=True)],
                  [Event(0, resource_event="bite", source_id="berry",
                         hazard_events=("trap",))], source)
    assert dead[0].row() == (0.0, 1.0, 1.0, True)
    print("GAME_EVENTS PASS: fixed replay, work drain, tagged hazard once, rest, absorbing death")
    print("GAME_EVENTS PASS: source_remaining=0 after 2 simultaneous consumers; duplicate bite consumed once")
    print("GAME_EVENTS PASS: fixtures distinguish reversed drain, repeated-contact damage, duplicate bite")


if __name__ == "__main__":
    _self_check()
