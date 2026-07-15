# VASP error triage & circumvention workflows

Companion to [`vasp_triage.py`](vasp_triage.py). Run it over a work
tree to get a per-directory diagnosis:

```bash
python3 vasp_triage.py /nobackup/pjalagam/CoCr_upstream --only-problems --fixes
python3 vasp_triage.py <WORK_ROOT> --json triage.json     # machine-readable
python3 vasp_triage.py <WORK_ROOT> --category electronic_scf
```

The signature catalog follows the community-standard list handled by
Materials Project's **custodian** `VaspErrorHandler`, plus completion /
truncation checks. Two practical notes baked into the scanner:

1. **Most VASP error strings go to STDOUT, not OUTCAR** (EDDDAV,
   ZBRENT, BRMIX, ZPOTRF…). ATAT wrappers capture stdout into
   `vasp.out` / `out.log` / `runstruct.log`; the scanner reads both
   OUTCAR-family files *and* those logs.
2. **A truncated OUTCAR is itself a diagnosis.** No
   `General timing and accounting` footer ⇒ the job was killed
   (walltime / OOM / crash) before VASP finished. For `OUTCAR.relax`,
   absence of `reached required accuracy` ⇒ the ionic loop never
   converged even if VASP exited cleanly.

---

## The workflows, per category

For each category: what it means, first-line fix, escalation, and how
to apply it **in this pipeline** (settings live in `vaspwrap.py`'s
`build_vasp_wrap` templates; relax behavior in `--relax-method`).

### 1. `electronic_scf` — SCF / diagonalization failures
`EDDDAV`, `EDDRMM/ZHEGV`, `ZHEEV`, `BRMIX: very serious problems`,
`Sub-Space-Matrix is not hermitian`, unconverged-at-NELM.

**Workflow:**
1. Delete `CHGCAR` + `WAVECAR` in the failing dir and rerun (stale
   wavefunctions after a geometry step cause most ZHEGV failures).
2. `ALGO = Normal` (from Fast) → if still failing `ALGO = All`
   (slowest, most robust). In `vasp.wrap`: change the `ALGO =` line
   (our `--algo` CLI flag drives this).
3. Metals with charge sloshing (BRMIX): `AMIX = 0.1`, `BMIX = 3.0`
   (damped Kerker mixing), `NELM = 120+`.
4. If BRMIX follows a run of increasingly weird geometries, the real
   problem is the relaxation — see category 2.

### 2. `ionic_relax` — relaxation algorithm failures
`ZBRENT: fatal error in bracketing`, `BRIONS problems: POTIM`,
positive TOTEN.

**Workflow:**
1. Restart from `CONTCAR` (copy → `POSCAR`) — most ZBRENT hits clear
   on a restart because the bracketing starts fresh.
2. Persisting ZBRENT: tighten `EDIFF = 1E-6` (forces are noisy when
   the SCF is loose) and/or switch `IBRION = 1` (quasi-Newton) after
   the geometry is roughly converged; keep `IBRION = 2` for the first
   steps.
3. `POTIM`: reduce to 0.25 when steps overshoot (ZBRENT/ZPOTRF after
   large cell changes), increase toward 0.5 on BRIONS complaints.
4. Pipeline lever: `--relax-method normal` (robustrelax) instead of
   `runstruct` — robustrelax has its own crash/restart tolerance loop
   and was built for exactly these flaky relaxations.
5. Positive TOTEN = broken structure. Don't tune INCAR; regenerate
   the geometry (check `str.out`, rerun the SQS generation).

### 3. `symmetry` — SYMPREC / rotation-matrix errors
`inv_rot_mat`, `SGRCON`, `PRICEL`, `POSMAP`, `RHOSYG`,
`point group operation missing`.

**Workflow (SQS-specific):** SQS supercells have *no* meaningful
symmetry; VASP's symmetry finder tripping over near-coincidences is
noise. Set `ISYM = 0` for all SQS runs (worth making the default in
`build_vasp_wrap` for this project) — that eliminates the whole
category. If you need symmetry on for endmembers, raise
`SYMPREC = 1e-4` instead.

### 4. `kpoints_tet` — tetrahedron / k-mesh failures
`Tetrahedron method fails`, `TETIRR`, `DENTET`, `Could not get correct
shifts`, `BZINTS`.

**Workflow:**
1. These fire when `ISMEAR = -5` meets a too-sparse or shifted mesh.
   For metals (our Co/Cr/Ni alloys): `ISMEAR = 1`, `SIGMA = 0.2` for
   relaxations; keep `-5` only for final static energies on meshes
   with ≥4 irreducible k-points.
2. Or densify: bump KPPRA a notch (our converge.py grid) so the mesh
   qualifies.
3. `Could not get correct shifts`: use a Γ-centred mesh.

### 5. `numerics_lapack` — LAPACK / projection breakdowns
`ZPOTRF`, `PSSYEVX`, `REAL_OPTLAY`, `RSPHER`.

**Workflow:**
1. `ZPOTRF` almost always means the cell degenerated mid-relax (atom
   pair distance → 0). Look at the last CONTCAR: if broken, restart
   from the last sane geometry with `POTIM = 0.25`, or pre-relax
   ions-only (`ISIF = 2`) before full `ISIF = 3`.
2. `REAL_OPTLAY` / `RSPHER`: set `LREAL = .FALSE.` (accurate
   projection; slower but stable — reasonable default for cells
   < ~30 atoms like ours).
3. `PSSYEVX`: `ALGO = Normal`, and if persisting, `NCORE = 1`.

### 6. `cell_basis` — cell/FFT sanity
`triple product of the basis vectors`, long-vector `AMIN` warning,
`aliasing errors`, FFT-grid insufficient.

**Workflow:**
1. Negative/zero triple product = left-handed or singular cell from
   structure generation — fix `str.out` (this is upstream of VASP).
2. Aliasing / FFT warnings: `PREC = Accurate` (already our default in
   the static wrap) or raise ENCUT.
3. `AMIN` warning on elongated cells: `AMIN = 0.01`.

### 7. `parallel_machine` — MPI / memory / launch
`MPI_ABORT`, `BAD TERMINATION`, OOM-kill, `ELF: KPAR>1`,
**`Problem running vasp command` / `unable to open OSZICAR`**.

**Workflow:**
1. `unable to open OSZICAR` right after launch = VASP never started.
   In this pipeline that's the missing MPI launcher: set
   `--cmd-prefix "mpiexec -n 128"` (CMD_PREFIX in the PBS template)
   and confirm the `comp-intel` + `mpi-intel` modules are loaded.
2. OOM: undersubscribe the node (e.g. `mpiexec -n 64` on 128 cores),
   or reduce `KPAR`/`NCORE`.
3. Generic `MPI_ABORT`: the true error is printed just above it in
   the log — triage that line, not the abort.

### 8. `incomplete_run` — killed before finishing
OUTCAR without the timing footer.

**Workflow:** check the PBS job epilogue for walltime/OOM. Resubmit
with a longer walltime, or restart from `CONTCAR` so the remaining
work fits. Our upstream pipeline's per-step logs
(`upstream_live.log` stage markers) tell you which stage the clock
ran out in.

### 9. `fitfc` unstable modes — not a VASP error, but you'll hunt it here
`Warning: <pert> is an unstable mode.` (dE<0 for a perturbation) and
`Unstable modes found.` + `Aborting.` in `fitfc_fit.log`. fitfc aborts
**before writing `svib_ht`** unless `-fn` or `-rl>0` is set.

**Workflow (automated by `phonon.run_fitfc`, policy
`--fitfc-on-unstable` / PBS `FITFC_ON_UNSTABLE`):**
1. `mark` (default): the SQS is left energy-only with the evidence in
   `<sqs>/unstable_modes.log`; downstream fits proceed without its
   vibrational term.
2. `escalate`: regenerate the perturbations at a 1.5× larger
   displacement radius and refit — imaginary modes that vanish were a
   finite-supercell artifact (short-range force constants extrapolated
   to Γ). Only the new `p*` dirs cost VASP time.
3. If the instability **persists**, it is likely genuine dynamical
   instability at that composition (classic for unstable endmember
   lattices, e.g. some BCC/SIGMA corners). Manual options, in order of
   preference: re-relax more tightly (`EDIFFG`, `--relax-method
   normal`) and rerun; `--fitfc-rl <len>` (van de Walle's robust
   soft-mode treatment, beta); `fitfc -fu` / `-gu=<n>` mode-following
   (produces a *different*, distorted structure — a scientific
   decision, not a pipeline one); or accept the SQS as energy-only.
4. `force` (`fitfc -fn`) is the last resort: the svib_ht it yields
   omits the unstable branches (lower bound) and is flagged as such.

---

## Automation options

- **Retry-with-fix loops:** [custodian](https://github.com/materialsproject/custodian)
  wraps VASP and applies exactly these fixes automatically
  (`VaspErrorHandler`), rewriting INCAR and restarting. Adopting it
  inside `runner.py` would replace manual triage for categories 1–5;
  the cost is taking a pymatgen dependency into the upstream tree.
- **ATAT-native:** `robustrelax_vasp` already implements a
  crash-tolerant relax loop (our `--relax-method normal`), which
  covers much of category 2 without new dependencies.
- **Pipeline gating:** rejected/failed runs never enter the fit anyway
  (energy-file gate + `--oszicar-min-score` + consensus target gate in
  `sqs2tdb_pipeline.py`) — triage is about *recovering* lost compute,
  not protecting the fit.

## Suggested routine

After every upstream PBS job:

```bash
python3 vasp_triage.py $WORK_ROOT --only-problems --fixes | tee triage_$(date +%Y%m%d).txt
python3 vasp_triage.py $WORK_ROOT --json triage.json
```

Then fix by category, biggest bucket first — one INCAR-policy change
(e.g. ISYM=0 for SQS) often clears dozens of directories at once.
