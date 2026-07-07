"""
upstream/ — first-principles calculation-generation half of the binary
TDB pipeline.

Where sqs2tdb_pipeline.py (one directory up) is the *downstream* consumer
that runs ``sqs2tdb -fit`` on finished DFT data, this package is the
*upstream* producer: it generates the SQS, runs ENCUT/KPPRA convergence
testing, drives structural relaxation (robustrelax or infdet), runs the
fitfc phonon workflow, and applies the DLM (disordered-local-moment)
bookkeeping needed before sqs2tdb_pipeline.py can fit anything.

Module map
----------
phases      Phase site/multiplicity constants, *_small single-sublattice
            systems, DLM spin conventions, SIGMA lev=3 handling config.
potcar      ENMAX extraction from POTCARs and ENCUT/KPPRA sweep grids.
vaspwrap    vasp.wrap (INCAR) generation for static / relax / phonon modes.
sqsgen      sqs2tdb -cp generation, *_small copy, randomspin, and the
            SIGMA lev=3 -> lev=0 +/-spin endmember conversion.
runner      Subprocess wrappers for pollmach / runstruct_vasp /
            robustrelax_vasp / infdet / fitfc with polling + logging.
converge    ENCUT + KPPRA convergence sweeps and 1 meV/atom selection.
relax       robustrelax (normal) or infdet structural relaxation.
phonon      fitfc phonon workflow + DLM str_relax.out/str_unpert.out fixup.
run_upstream  CLI orchestrator chaining all of the above per phase/binary.
"""

__all__ = [
    "phases",
    "potcar",
    "vaspwrap",
    "sqsgen",
    "runner",
    "converge",
    "relax",
    "phonon",
]
