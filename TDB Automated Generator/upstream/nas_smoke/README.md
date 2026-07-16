# nas_smoke — VASP plumbing smoke suite

Five fast tests, one per **distinct VASP call path** the upstream
pipeline uses. Run this before any full upstream job: it costs minutes
on one `devel`-queue node and tells you exactly which link of the
ATAT→VASP chain is broken, instead of a 72-hour job dying at hour 3.

| Test | Call | Production path it covers | Pass criteria |
|---|---|---|---|
| `T1_static` | `runstruct_vasp <launcher>` + static `vasp.wrap` | `converge.run_static_point` — every ENCUT/KPPRA sweep point (the historical "unable to open OSZICAR" site) | `energy`, `str_relax.out` |
| `T2_runstruct` | `runstruct_vasp <launcher>` + relax wrap (NSW=5) | `--relax-method runstruct` relaxation + extraction | `str_relax.out`, `energy`, `force.out` |
| `T3_robustrelax` | `robustrelax_vasp -mk` then `robustrelax_vasp -id -c 0.05 <launcher>` | `--relax-method infdet` (the default: inflection detection with its required strain cutoff); early-stopped via the `stop` sentinel once `str_relax.out` exists | `str_relax.out` |
| `T4_fitfc_wrap` | `runstruct_vasp -w fvasp.wrap <launcher>` on a displaced frozen cell | fitfc perturbation force runs (separate wrap file, NSW=0, forces → `force.out`) | `force.out`, `str_relax.out` |
| `T5_pollmach` | `pollmach runstruct_vasp <launcher>` over two `wait`-marked subdirs | the dispatcher every stage routes through (wait consumption, walk-up `vasp.wrap`, `stoppoll` shutdown) | `p_1/energy`, `p_2/energy` |

All tests use one 2-atom cell, ENCUT=300, KPPRA=1000, NELM≤25, NSW≤5,
NCORE=1/KPAR=1, and **spin off** — this suite validates plumbing, not
physics (production auto-enables ISPIN=2 for magnetic elements).

## Run it

```bash
# edit the USER CONFIG block, then:
qsub submit_smoke.pbs
```

or directly on a node with modules loaded:

```bash
python3 run_smoke.py --element Co \
    --potcar $VASP_PP/Co/POTCAR \
    --cmd-prefix "mpiexec -n 8" --env-bin $HOME/bin \
    --only T1_static            # subset selection when re-testing one path
```

`--dry-run` builds all case directories and `plan.json` (the exact argv
per test) without launching anything — useful to eyeball the inputs.

## Resource footprint

One node, `devel` queue, 30-minute walltime cap, 8 MPI ranks. Every
test self-terminates through its ATAT stop sentinel the moment the
expected outputs exist, plus a hard per-test `--timeout` (default
1200 s in the driver, 900 s in the PBS config). A healthy suite is
typically done in well under 10 minutes.

## Reading the results

- `smoke_report.txt` — PASS/FAIL per test with elapsed time, energy,
  missing files, and the tail of the decisive log.
- `smoke_report.json` — the same plus full log tails and the preflight
  (binary paths, launcher check, POTCAR mechanism). **This is the file
  to paste back when asking for debugging help.**
- `environment.txt` (PBS wrapper) — `module list`, `which` for every
  binary, and a 2-rank `mpiexec hostname` sanity check.
- `triage.json` — `vasp_triage.py` scan of the whole smoke tree
  (custodian-style error signatures + suggested fixes per category).

Typical failure signatures:

| Symptom | Meaning | Fix |
|---|---|---|
| preflight `missing_binaries` | ATAT not on PATH | check `ATAT_BIN`, modules |
| every test: `unable to open OSZICAR` in log tail | VASP launched bare (MPI build) | set `CMD_PREFIX="mpiexec -n 8"`, check `mpi-intel` module (see `environment.txt` MPI sanity line) |
| T1 fails, others too, `POTCAR` errors in log | POTCAR mechanism broken | pass `--potcar`, or fix `POTDIR` in `~/.ezvasp.rc` |
| T4 alone fails | `-w` wrap-name path broken / frozen-run extraction | check `fvasp.wrap` was read (grep `T4_fitfc_wrap/INCAR` for `NSW = 0`) |
| T5 alone fails | pollmach dispatch (wait files, walk-up wrap) | check `T5_pollmach/pollmach.log`; ensure `stoppoll` isn't pre-existing |
| PASS but slow / timeout kills | node/queue contention or NELM too low to converge SCF | raise `--timeout`; NELM non-convergence still writes outputs (rc recorded) |
