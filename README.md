# IM2Deep
Collisional cross-section prediction for (modified) peptides.

---
## Introduction

IM2Deep is a CCS predictor for (modified) peptides.
It is able to accurately predict CCS for modified peptides, even if the modification wasn't observed during training.

## Installation
Install with pip:

`pip install im2deep`

If you want to use the multi-output model for CCS prediction of multiconformational peptide ions, use the following installation command:

`pip install 'im2deep[er]'`

## Usage
### Basic CLI usage:
```sh
im2deep <path/to/peptide_file.csv>
```
If you want to calibrate your predictions (HIGHLY recommended), please provide a calibration file:
```sh
im2deep <path/to/peptide_file.csv> --calibration-file <path/to/peptide_file_with_CCS.csv>
```
To use the multi-output prediction model on top of the original model, provide the -e flag 
(make sure you have the optional dependencies installed!):
```sh
im2deep <path/to/peptide_file.csv> --calibration-file <path/to/peptide_file_with_CCS.csv> -e
```
For an overview of all CLI arguments, run `im2deep --help`.

## Input files
Both peptide and calibration files are expected to be comma-separated values (CSV) with the following columns:
  - `seq`: unmodified peptide sequence
  - `modifications`: every modifications should be listed as `location|name`, separated by a pipe character (`|`)
     between the location, the name, and other modifications. `location` is an integer counted starting at 1 for the
     first AA. 0 is reserved for N-terminal modifications, -1 for C-terminal modifications. `name` has to correspond
     to a Unimod (PSI-MS) name.
  - `charge`: peptide precursor charge
  - `CCS`: collisional cross-section (only for calibration file)

For example:

```csv
seq,modifications,charge,CCS
VVDDFADITTPLK,,2,422.9984309464991
GVEVLSLTPSFMDIPEK,12|Oxidation,2,464.6568644356109
SYSGREFDDLSPTEQK,,2,468.9863221739147
SYSQSILLDLTDNR,,2,460.9340710819608
DEELIHLDGK,,2,383.8693416055445
IPQEKCILQTDVK,5|Butyryl|6|Carbamidomethyl,3,516.2079366048176
```

## Citing
If you use IM2Deep within the context of [(TI)MS<sup>2</sup>Rescore](https://github.com/compomics/ms2rescore), please cite the following:
> **TIMS²Rescore: A DDA-PASEF optimized data-driven rescoring pipeline based on MS²Rescore.**
> Arthur Declercq*, Robbe Devreese*, Jonas Scheid, Caroline Jachmann, Tim Van Den Bossche, Annica Preikschat, David Gomez-Zepeda, Jeewan Babu Rijal, Aurélie Hirschler, Jonathan R Krieger, Tharan Srikumar, George Rosenberger, Dennis Trede, Christine Carapito, Stefan Tenzer, Juliane S Walz, Sven Degroeve, Robbin Bouwmeester, Lennart Martens, and Ralf Gabriels.
> _Journal of Proteome Research_ (2025) [doi:10.1021/acs.jproteome.4c00609](https://doi.org/10.1021/acs.jproteome.4c00609) <span class="__dimensions_badge_embed__" data-doi="10.1021/acs.jproteome.4c00609" data-hide-zero-citations="true" data-style="small_rectangle"></span>

In other cases, please cite the following:
> **Collisional cross-section prediction for multiconformational peptide ions with IM2Deep.**
> Robbe Devreese, Alireza Nameni, Arthur Declercq, Emmy Terryn, Ralf Gabriels, Francis Impens, Kris Gevaert, Lennart Martens, Robbin Bouwmeester.
> _Anal. Chem._ (2025) [doi:10.1021/acs.analchem.5c01142](https://pubs.acs.org/doi/10.1021/acs.analchem.5c01142)



