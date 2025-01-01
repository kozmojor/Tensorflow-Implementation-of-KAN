# README for KAN Reproduction Project by KANY group, ECBM 4040.

## Overview
This repository contains the project files for implementation and reproduction of Kolmogorov-Arnold Networks (KAN) in Tensorflow, an advanced neural network architecture inspired by the Kolmogorov-Arnold Representation Theorem. You can find the original work here ([Kolmogorov–Arnold Networks paper](https://arxiv.org/abs/2010.03645)). The repository includes all the files and scripts necessary to perform the experiments, reproduce the results, and explore the theoretical aspects of KAN. For detailed explanations, the project report (`E4040.2024Fall.KANY.report.hd2573.jx2598.yk3108.pdf`) provides comprehensive insights into the methodology and results.

![image](https://github.com/user-attachments/assets/b0e996da-894c-483d-a8cc-112cdc4fb079)

## Getting Started

### Prerequisites
Ensure you have the required dependencies installed. These can be found in the `requirement.txt` file. Install them using the following command:
```bash
pip install -r requirement.txt
```

### Running the Experiments
This repository contains multiple Jupyter Notebooks (`.ipynb`) that correspond to various experiments related to KAN:

1. **Ex1.KAN_Pipeline.ipynb**:
   - Demonstrates the core pipeline for implementing KAN and provides an overview of the workflow.

2. **Ex2.Grid_Refine.ipynb**:
   - Conducts experiments on grid refinement, showing how finer grids improve approximation accuracy.

3. **Ex3.Deep_Kan.ipynb**:
   - Explores deeper KAN architectures, evaluating performance with additional layers.

4. **Ex4.Classfication.ipynb**:
   - Applies KAN to classification tasks, demonstrating its adaptability to supervised learning.

5. **Ex5.Special_Functions.ipynb**:
   - Handles complex special functions using KAN, including non-standard cases.

6. **Ex6.PDE_interpretation.ipynb**:
   - Interprets partial differential equations (PDEs) using KAN-based models.

7. **Ex7.PDE_accuracy.ipynb**:
   - Evaluates the accuracy of KAN in solving PDEs, comparing it to benchmarks.

8. **Ex8.continue_learning.ipynb**:
   - Tests continual learning capabilities, showing how KAN adapts to new data.

9. **Ex9.singularity.ipynb**:
   - Addresses singularities and examines KAN's robustness in challenging scenarios.

10. **Ex10.Comparison.ipynb**:
    - Compares KAN with other machine learning approaches to highlight its strengths.

![image](https://github.com/user-attachments/assets/ef5014ec-484f-408c-8ced-00ffff714d0b)

To start, run the Jupyter Notebook of interest by navigating to the directory and executing:
```bash
jupyter notebook <notebook_name>.ipynb
```

## Key Components

- **Core Modules**: The `tensorkan/` and `KERAS_KAN/kan/` directories contain the core implementation of KAN. These include:

  - `KANLayer.py`: Defines the main KAN layer.
  - `spline.py`: Contains spline evaluation and conversion utilities.
  - `utils.py`: Provides helper functions for dataset preparation and grid management.
  - `LBFGS.py`: Implements the LBFGS optimization algorithm for efficient model training.
  - `MultKAN.py`: Supports multi-layer KAN model construction and training.
  - `Symbolic_KANLayer.py`: Enables symbolic KAN representation for interpretability.

- **Figures and Visualizations**: The `figures/` and `img/` directories store images and figures used in the report and visualizations, such as grid refinements and spline functions.

- **Report**: The `KANY_report.pdf` contains detailed explanations, experimental results, and insights. The `results` section maps directly to the experiments in the provided `.ipynb` files.


## Repository Structure
```
repo/
├── .git                   # Git-related metadata
├── CHANGELOG.md           # Project change history
├── Ex1.KAN_Pipeline.ipynb # Pipeline for KAN implementation
├── Ex2.Grid_Refine.ipynb  # Grid refinement experiments
├── Ex3.Deep_Kan.ipynb     # Exploration of deep KAN architectures
├── Ex4.Classfication.ipynb# Classification experiments with KAN
├── Ex5.Special_Functions.ipynb # Handling special functions with KAN
├── Ex6.PDE_interpretation.ipynb# PDE interpretation using KAN
├── Ex7.PDE_accuracy.ipynb # PDE accuracy evaluation with KAN
├── Ex8.continue_learning.ipynb # Experiments on continual learning
├── Ex9.singularity.ipynb  # Addressing singularities in KAN
├── Ex10.Comparison.ipynb  # Comparative study of KAN vs other methods
├── figures/               # Figures generated by KAN model
├── img/                   # Additional images for documentation
│   ├── Grid.png
│   ├── Spline.png
│   ├── ...
├── KANY_report.pdf        # Comprehensive project report
├── KERAS_KAN/             # Implementation using Keras
│   ├── kan/               # Core KAN modules
│   │   ├── KANLayer.py    # Definition of KAN layers
│   │   ├── spline.py      # Spline computation functions
│   │   ├── utils.py       # Utility functions for KAN
├── README.md              # Project documentation
├── ReportPlot.ipynb       # Visualization scripts
├── requirement.txt        # Project dependencies
├── structure.txt          # Directory structure
├── tensorkan/             # TensorFlow implementation of KAN
│   ├── KANLayer.py
│   ├── LBFGS.py           # LBFGS optimization algorithm
│   ├── MultKAN.py         # Multi-layer KAN model
│   ├── spline.py
│   ├── Symbolic_KANLayer.py
│   ├── utils.py
```

## Contributing
Contributions to improve this repository are welcome. Please fork the repository and submit a pull request for review.


## License
This project is licensed under the MIT License. 

---

This repository provides all necessary files for implementing and understanding KAN. For further questions, refer to the `E4040.2024Fall.KANY.report.hd2573.jx2598.yk3108.pdf` or contact the authors listed in the report.
