# Physics-Informed Neural Networks - Workshop for ML4SCIENCE & ENGINEERING SUMMER SCHOOL by @inductiva

This repository contains a hands-on workshop on implementing Physics-Informed Neural Networks (PINNs) in PyTorch for the [ML4SCIENCE & ENGINEERING SUMMER SCHOOL](https://inductiva.ai/events/machine-learning-summer-school), organized by Inductiva.ai @FEUP. You will solve both a forward problem (Burgers' equation) and an inverse problem (1D Euler equations) via Jupyter notebooks.

> Adapted from TUM Physics-Informed Machine Learning lab session on "Physics-Informed Machine Learning", @tummfm

## Repository Structure

```
├── data/           # Raw data files used by the notebooks
│   ├── burgers_shock.mat
│   └── 1DEuler_data.npy
├── to_do/           # Notebooks for you to complete
│   ├── 1-forward_pinn.ipynb
│   └── 2-inverse_pinn.ipynb
├── solved/         # Completed solution notebooks
│   ├── 1-forward_pinn.ipynb
│   └── 2-inverse_pinn.ipynb
└── README.md       # This file
```

* **data/**: Contains all necessary datasets (`.mat` and `.npy`) for the exercises. These files are loaded by the notebooks.
* **todo/**: Contains starter notebooks with scaffolding and instructions. Your tasks are marked within code cells and markdown sections.
* **solved/**: Provides fully worked-out solutions, including plotting and analysis. Use these to check your work.

## Prerequisites

* **Python 3.12+**
* **Git**
* **Visual Studio Code** with the following extensions:

  * Python (ms-python.python)
  * Jupyter (ms-toolsai.jupyter)

Optional but recommended:

* **Anaconda** or **Miniconda** for environment management.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/biromiro/pinn-workshop-aptwind.git
   cd pinn-workshop-aptwind
   ```

2. **Create and activate a virtual environment**

   Using `venv`:

   ```bash
   python -m venv venv
   # On Linux/Mac
   source venv/bin/activate
   # On Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```

   Or using Conda:

   ```bash
   cconda create --name pinn-workshop
   conda activate pinn-workshop
   ```

3. **Install dependencies**

   Otherwise, install packages directly:

   ```bash
   pip install torch numpy scipy matplotlib tqdm jupyter ipywidgets widgetsnbextension
   ```

   or, with conda:

   ```bash
   conda install pytorch numpy scipy matplotlib tqdm jupyter ipywidgets widgetsnbextension
   ```

## Running the Notebooks in VS Code

1. **Open the folder**

   * Launch VS Code and select **File → Open Folder...**, then choose this repository.

2. **Select the Python interpreter**

   * Press `Ctrl+Shift+P`, type `Python: Select Interpreter`, and choose the interpreter from your `venv` or Conda environment.

3. **Open a notebook**

   * Navigate to the `todo/` folder and open one of the `.ipynb` files.

4. **Run cells**

   * Click the **Run All** button in the top toolbar to execute every cell in order. Alternatively, run cells individually using the ▶️ icons.

5. **Switch to solution**

   * After working through `to_do/`, compare your results with the solution in the corresponding notebook under `solved/`.

## Tips

* **GPU Support**: The notebooks detect CUDA or Apple MPS devices automatically. Ensure your system has a supported GPU and drivers.
* **Loss Monitoring**: Look at training-loss plots to check convergence. You can adjust learning rates or network architectures as an exercise.

## License

This workshop content is provided under the MIT License. See [LICENSE](LICENSE) for details.
