# Changelog

Full notes in `docs/releases/`. Newest first. Format: `~/iTools/dev/release-note/TEMPLATE.md`.

- [v0.12.1](docs/releases/v0.12.1.md) (2026-10-26): muscle GPU builds no longer overrun host memory; a 65,536-body run needs 1.99 GB host RAM instead of 9.27 GB
- [v0.12.0](docs/releases/v0.12.0.md) (2026-10-23): thirteen former MuJoCo refusals ported and gated (friction loss, condim 4/6, PGS/CG over every row, margin-0 mesh-mesh, ...); living-world renderer; muscle engine; streamed heightfield
- [v0.11.0](docs/releases/v0.11.0.md) (2026-10-12): motion dataset driver (own clip, push, real fall per animal), per-animal eq_active
- [v0.10.0](docs/releases/v0.10.0.md) (2026-10-12): one canonical body for any mocap skeleton, spike repair, limb planting
- [v0.9.0](docs/releases/v0.9.0.md) (2026-10-12): watch any body live: Viewer and view.mojo (check.sh view)
- [v0.8.0](docs/releases/v0.8.0.md) (2026-10-12): mocap clips drive a physics character (bvh, track, motion metrics)
- [v0.7.0](docs/releases/v0.7.0.md) (2026-10-12): mesh-mesh contact (mjc_Convex, check.sh meshmesh); margin-0 box/mesh pairs refused
- [v0.6.0](docs/releases/v0.6.0.md) (2026-10-12): push input (MuJoCo xfrc_applied, sparse slots, check.sh xfrc); docs/API.md
- [v0.5.0](docs/releases/v0.5.0.md) (2026-10-12): pairs, tendons and equalities across trees in one model (check.sh cross); endless city (city.mojo, render -DCITY=1); world assets in Mojo
- [v0.4.0](docs/releases/v0.4.0.md) (2026-10-09): live brain view (brain_view.mojo, 96 us for 138,639 neurons); the engine window's GUI moved to mojo-imkit
- [v0.3.0](docs/releases/v0.3.0.md) (2026-10-09): The fly looks like MuJoCo\'s: textures, see-through wings, reflections, the arena; smooth multi-fly view.
- [v0.2.0](docs/releases/v0.2.0.md) (2026-10-09): Solid-looking bodies: smooth shading, sun shadows, contact occlusion, smooth edges; about 1 % of a live frame.
- [v0.1.0](docs/releases/v0.1.0.md) (2026-10-09): First release as its own repo: all 35 stages proven from here, CPU smoke, 18x shorter GPU hold for the full gate.
