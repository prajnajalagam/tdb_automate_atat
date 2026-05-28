# README for plot_infdet.py

## Overview
`plot_infdet.py` is a Python script designed to assist users working with the inflection detection feature, specifically the `robustrelax_vasp -id` mode of the Alloy Theoretic Automated Toolkit (ATAT). This script is particularly useful when runs do not fully terminate or converge. It provides a visualization of how the minimum curvature (`mincurv`) varies with the gradient (`grad`), enabling users to make an informed decision about the "best" energy choice in mechanically unstable scenarios.

## Features
- Visualizes the relationship between `mincurv` and `grad`.
- Helps identify the most appropriate energy value when runs are not well converged.
- Simplifies the analysis of mechanically unstable configurations.

## Prerequisites
- Python 3.x
- Required Python libraries:
    - `matplotlib`
    - `numpy`
    - Any other dependencies (install as needed)

## Usage
1. Ensure that the `plot_infdet.py` script is in the same directory as your data files or adjust the paths accordingly.
2. Prepare the input data generated from the `robustrelax_vasp -id` mode of ATAT.
3. Run the script using the command:
     ```
     python plot_infdet.py
     ```
4. The script will generate a plot showing the variation of `mincurv` with `grad`.

## Output
The script produces a graphical plot that allows users to visually analyze the data and make an informed decision about the energy value to use in cases of mechanical instability.

## Notes
- Ensure that the input data is correctly formatted and complete for accurate visualization.
- This script is intended for use in scenarios where runs are not fully converged and additional analysis is required.

## License
This script is provided as-is. Please ensure compliance with any relevant licensing terms for ATAT and associated tools.

## Author
Prajna Jalagam, Brown University and NASA Ames Research Center

## Acknowledgments
Special thanks to the developers of ATAT for providing robust tools for alloy theory and related calculations.