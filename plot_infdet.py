#!/usr/bin/env python3
"""
Plot energy, mincurv, and grad_norm from infdet.log on a single figure.

Usage:
  python /home7/pjalagam/scripts/plot_infdet.py           # plots all */infdet.log
  python /home7/pjalagam/scripts/plot_infdet.py 01        # plots 01/infdet.log only
  python /home7/pjalagam/scripts/plot_infdet.py 01 03 05  # plots selected subdirs

Plots are saved as .png files in each subdirectory.
"""

import sys
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_dirs_with_infdet(base):
    """Find all immediate subdirectories containing infdet.log."""
    dirs = []
    for entry in sorted(os.listdir(base)):
        candidate = os.path.join(base, entry)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, 'infdet.log')):
            dirs.append(entry)
    return dirs


def plot_infdet(dirname):
    """Parse and plot infdet.log in the given directory."""
    logfile = os.path.join(dirname, 'infdet.log')

    if not os.path.isfile(logfile):
        print(f"  Skipping {dirname}: infdet.log not found.")
        return

    mincurv, energy, grad_norm = [], [], []
    pattern = re.compile(
        r'mincurv=\s*([-\d.]+)\s+energy=\s*([-\d.]+)\s+grad_norm=\s*([-\d.]+)'
    )

    with open(logfile) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                mincurv.append(float(m.group(1)))
                energy.append(float(m.group(2)))
                grad_norm.append(float(m.group(3)))

    if not energy:
        print(f"  Skipping {dirname}: no 'mincurv' lines found.")
        return

    step = list(range(len(energy)))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 5), sharex=True)

    # Top panel: Energy
    ax1.plot(step, energy, 'o-', color='#1f77b4', markersize=3)
    ax1.set_ylabel('Energy (eV)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Middle panel: Min. Curvature
    ax2.plot(step, mincurv, 's-', color='#d62728', markersize=3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Min. Curvature', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Bottom panel: Gradient Norm
    ax3.plot(step, grad_norm, '^-', color='#2ca02c', markersize=3)
    ax3.set_ylabel('Gradient Norm', fontsize=12)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'{dirname}: Inflection Detection Convergence', fontsize=14)
    fig.tight_layout()
    outpath = os.path.join(dirname, 'infdet_convergence.png')
    fig.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"  {dirname}: {len(energy)} iterations, final energy = {energy[-1]:.3f}, saved {outpath}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dirs = sys.argv[1:]
    else:
        dirs = find_dirs_with_infdet('.')
        if not dirs:
            print("No subdirectories with infdet.log found in current directory.")
            sys.exit(1)
        print(f"Found {len(dirs)} directories: {', '.join(dirs)}")

    for d in dirs:
        plot_infdet(d)