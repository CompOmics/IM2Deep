# IM2Deep
Collisional cross-section prediction for (modified) peptides.

---
## Introduction

IM2Deep is a deep learning-based CCS predictor for (modified) peptides. It accurately predicts collisional cross-section (CCS) values for modified peptides, even if the modification wasn't observed during training. The tool supports both single-conformer and multi-conformer predictions for peptide ions.

## Installation

With Python 3.11 or higher, install with pip:

```bash
pip install im2deep
```

We recommend using a [venv](https://docs.python.org/3/library/venv.html) or
[conda](https://docs.conda.io/en/latest/) virtual environment.

### For development

Clone this repository and use [uv](https://docs.astral.sh/uv/) to install:

```bash
git clone https://github.com/CompOmics/IM2Deep.git
cd IM2Deep
uv sync --group dev --group docs
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

### Python API

IM2Deep can also be used programmatically:

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

```text
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
If you use IM2Deep within the context of [(TI)MS²Rescore](https://github.com/compomics/ms2rescore), please cite the following:
> **TIMS²Rescore: A DDA-PASEF optimized data-driven rescoring pipeline based on MS²Rescore.**
> Arthur Declercq*, Robbe Devreese*, Jonas Scheid, Caroline Jachmann, Tim Van Den Bossche, Annica Preikschat, David Gomez-Zepeda, Jeewan Babu Rijal, Aurélie Hirschler, Jonathan R Krieger, Tharan Srikumar, George Rosenberger, Dennis Trede, Christine Carapito, Stefan Tenzer, Juliane S Walz, Sven Degroeve, Robbin Bouwmeester, Lennart Martens, and Ralf Gabriels.
> _Journal of Proteome Research_ (2025) [doi:10.1021/acs.jproteome.4c00609](https://doi.org/10.1021/acs.jproteome.4c00609) <span class="__dimensions_badge_embed__" data-doi="10.1021/acs.jproteome.4c00609" data-hide-zero-citations="true" data-style="small_rectangle"></span>

In other cases, please cite the following:
> **Collisional cross-section prediction for multiconformational peptide ions with IM2Deep.**
> Robbe Devreese, Alireza Nameni, Arthur Declercq, Emmy Terryn, Ralf Gabriels, Francis Impens, Kris Gevaert, Lennart Martens, Robbin Bouwmeester.
> _Anal. Chem._ (2025) [doi:10.1021/acs.analchem.5c01142](https://pubs.acs.org/doi/10.1021/acs.analchem.5c01142)


