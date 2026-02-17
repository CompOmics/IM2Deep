# IM2Deep
Collisional cross-section prediction for (modified) peptides.

---
## Introduction

IM2Deep is a deep learning-based CCS predictor for (modified) peptides. It accurately predicts collisional cross-section (CCS) values for modified peptides, even if the modification wasn't observed during training. The tool supports both single-conformer and multi-conformer predictions for peptide ions.

## Installation
Install with pip:

```bash
pip install im2deep
```

## Usage

### Command Line Interface (CLI)

**Basic prediction:**
```bash
im2deep <path/to/peptide_file.csv>
```

**With calibration (HIGHLY recommended):**
```bash
im2deep <path/to/peptide_file.csv> --calibration-precursors <path/to/calibration_file.csv>
```

**Calibration options:**
- `--calibrate-per-charge`: Calculate separate calibration shift factors per charge state (recommended, default true)
- `--use-charge-state`: Charge state for global calibration when --calibrate-per-charge is disabled

**Multi-conformer prediction:**
To use the multi-output prediction model (requires optional dependencies):
```bash
im2deep <path/to/peptide_file.csv> --calibration-precursors <path/to/calibration_file.csv> --multi
```

**Output options:**
```bash
im2deep <path/to/peptide_file.csv> --output-file predictions.csv
```

For a complete overview of all CLI arguments, run:
```bash
im2deep --help
```

### Training Custom Models

IM2Deep now includes integrated training functionality for creating custom models on your own datasets:

**Training a new model:**
```bash
im2deep train <path/to/config.json>
```

The training configuration file should be in JSON format and include:
- `data_path`: Path to training data (CSV or pickle format)
- `output_path`: Directory where the model and checkpoints will be saved
- `model_params`: Model architecture and training parameters
  - `model_name`: Name for the model
  - `epochs`: Number of training epochs
  - `batch_size`: Training batch size
  - `learning_rate`: Learning rate for optimization
  - `device`: GPU device index (e.g., 0)
  - `monitor`: Metric to monitor for checkpointing (e.g., "val_loss")
  - `mode`: "min" or "max" for checkpoint metric
  - `use_best_model`: Whether to use the best checkpoint
  - `multi-output`: Enable multi-conformer prediction training
  - `transfer`: Use transfer learning from pre-trained backbone
  - `wandb`: Weights & Biases logging configuration
    - `enabled`: Enable W&B logging
    - `project_name`: W&B project name
- `test_split`: Fraction of data to use for testing (e.g., 0.1)
- `val_split`: Fraction of training data to use for validation (e.g., 0.1)
- `remove_charge_dupes`: Remove duplicate sequences with different charges
- `save_dfs`: Save train/val/test dataframes
- `save_data_tensors`: Save preprocessed data tensors

**Example config.json:**
```json
{
  "data_path": "training_data.csv",
  "output_path": "./output",
  "test_split": 0.1,
  "val_split": 0.1,
  "remove_charge_dupes": true,
  "save_dfs": false,
  "save_data_tensors": false,
  "model_params": {
    "model_name": "my_custom_model",
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.001,
    "device": 0,
    "monitor": "Validation MAE",
    "mode": "min",
    "use_best_model": true,
    "multi-output": false,
    "transfer": false,
    "add_X_mol": false,
    "wandb": {
      "enabled": false,
      "project_name": "im2deep-training"
    }
  }
}
```

The training data should be in CSV format with columns: `seq`, `modifications`, `charge`, and `CCS`. For multi-conformer training, `CCS` should contain a list of two values.

### Python API

IM2Deep can also be used programmatically:

**Prediction:**
```python
from im2deep import predict, predict_and_calibrate
from psm_utils import PSMList

# Load your peptides as PSMList
psm_list = PSMList(psm_list=[...])  # or use psm_utils.io.read_file()

# Simple prediction
predictions = predict(psm_list)

# Prediction with calibration
psm_list_calibration = PSMList(psm_list=[...])  # Must contain CCS values
calibrated_predictions = predict_and_calibrate(
    psm_list=psm_list,
    psm_list_cal=psm_list_calibration
)
```

**Training:**
```python
from im2deep.training_data import data_extraction
from im2deep.training import train_model
from im2deep.training_evaluate import evaluate_and_plot

# Prepare configuration dictionary (same structure as JSON config)
config = {
    "data_path": "training_data.csv",
    "output_path": "./output",
    "test_split": 0.1,
    "val_split": 0.1,
    "model_params": {
        "model_name": "my_model",
        "epochs": 100,
        "batch_size": 256,
        # ... other parameters
    }
}

# Extract and prepare data
data, test_df = data_extraction(config)

# Train the model
trainer, model, test_loader = train_model(
    data, 
    config["model_params"], 
    output_path=config["output_path"]
)

# Evaluate and visualize results
evaluate_and_plot(trainer, model, test_loader, test_df, config)
```

## Input Files

### Standard Format
IM2Deep accepts any format supported by [psm_utils](https://github.com/compomics/psm_utils), including:
- Peptide Record (.peprec)
- MaxQuant msms.txt
- MSFragger PSM files
- And more...

### Legacy CSV Format
Alternatively, use comma-separated values (CSV) with the following columns:

- **`seq`**: Unmodified peptide sequence
- **`modifications`**: Modifications listed as `location|name`, separated by pipe (`|`) characters
  - `location`: Integer starting at 1 for the first amino acid
    - `0` = N-terminal modification
    - `-1` = C-terminal modification
  - `name`: Must correspond to a Unimod (PSI-MS) name
- **`charge`**: Peptide precursor charge state
- **`CCS`**: Collisional cross-section (only required for calibration files)

**Example:**

```csv
seq,modifications,charge,CCS
VVDDFADITTPLK,,2,422.9984309464991
GVEVLSLTPSFMDIPEK,12|Oxidation,2,464.6568644356109
SYSGREFDDLSPTEQK,,2,468.9863221739147
SYSQSILLDLTDNR,,2,460.9340710819608
DEELIHLDGK,,2,383.8693416055445
IPQEKCILQTDVK,5|Butyryl|6|Carbamidomethyl,3,516.2079366048176
```

## Important Notes

- **Calibration**: Highly recommended for accurate predictions. Calibration corrects for systematic differences between predicted and observed CCS values.
- **Charge states**: IM2Deep predictions are reliable for charge states up to z=6. PSMs with higher charge states are automatically filtered out during validation.

## Citing
If you use IM2Deep within the context of [(TI)MS<sup>2</sup>Rescore](https://github.com/compomics/ms2rescore), please cite the following:
> **TIMS²Rescore: A DDA-PASEF optimized data-driven rescoring pipeline based on MS²Rescore.**
> Arthur Declercq*, Robbe Devreese*, Jonas Scheid, Caroline Jachmann, Tim Van Den Bossche, Annica Preikschat, David Gomez-Zepeda, Jeewan Babu Rijal, Aurélie Hirschler, Jonathan R Krieger, Tharan Srikumar, George Rosenberger, Dennis Trede, Christine Carapito, Stefan Tenzer, Juliane S Walz, Sven Degroeve, Robbin Bouwmeester, Lennart Martens, and Ralf Gabriels.
> _Journal of Proteome Research_ (2025) [doi:10.1021/acs.jproteome.4c00609](https://doi.org/10.1021/acs.jproteome.4c00609) <span class="__dimensions_badge_embed__" data-doi="10.1021/acs.jproteome.4c00609" data-hide-zero-citations="true" data-style="small_rectangle"></span>

In other cases, please cite the following:
> **Collisional cross-section prediction for multiconformational peptide ions with IM2Deep.**
> Robbe Devreese, Alireza Nameni, Arthur Declercq, Emmy Terryn, Ralf Gabriels, Francis Impens, Kris Gevaert, Lennart Martens, Robbin Bouwmeester.
> _Anal. Chem._ (2025) [doi:10.1021/acs.analchem.5c01142](https://pubs.acs.org/doi/10.1021/acs.analchem.5c01142)


