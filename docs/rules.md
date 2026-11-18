# Rules

Each line: what broke -> rule.

- `perl -pi`/`sed -i` over `git ls-files` turned a symlink into a copy of its target -> drop mode 120000 entries (`git ls-files -s`) before any in-place rewrite; an unbalanced +/- diff stat is the alarm.
- A mutant passed because the gate broke on a missing test body -> `tools/bin/mutant-check` fails on missing-file/module errors; every mutant gate takes the same `BODY` as its main gate.
- `mj-oracle` moved out of `~/iTools` and every build broke -> no check depends on a path outside the repo; vendor or fall back (`tools/bin/`).
