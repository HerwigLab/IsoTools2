[![tests](https://github.com/HerwigLab/IsoTools2/actions/workflows/tests.yml/badge.svg)](https://github.com/HerwigLab/IsoTools2/actions?query=workflow%3Atests)
[![docs](https://readthedocs.org/projects/isotools/badge/?version=latest)](https://isotools.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/isotools.svg)](https://pypi.org/project/isotools)
[![PyPIDownloadsTotal](https://pepy.tech/badge/isotools)](https://pepy.tech/project/isotools)
[![Licence: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/HerwigLab/IsoTools2/blob/master/LICENSE.txt)
<img align="right" src="IsoToolsLogo.png" alt="IsoTools Logo" width="300"  />

# IsoTools

IsoTools is a python module for Long Read Transcriptome Sequencing (LRTS) analysis.

Key features:

* Import of LRTS bam files (aligned full length transcripts).
* Import of reference annotation in gff3/gtf format.
* Computation of quality control metrics.
* Annotation and classification of novel transcripts using the biologically motivated classification scheme SQANTI.
* Evaluation of the coding potential of isoforms.
* Definition of alternative splicing events based on segment graphs.
* Detection of differential alternative splicing between samples and groups of samples.
* Gene modelling based on structural and expression variability.
* Support for proteogenomic approaches at the interface of transcriptomics and proteomics.
* Various data visualizations.

## Documentation

The documentation, including tutorials with real-world case studies and the complete API reference is available at [readthedocs](https://isotools.readthedocs.io/en/latest/ "documentation")

## Installation

Isotools is available from PyPI, and can be installed with the pip command:

```bash
python3 -m pip install isotools
```

Alternatively, to install from github, use the following command:

```bash
git clone https://github.com/HerwigLab/IsoTools2.git
cd isotools
python3 -m pip install .
```

## Usage

This code block demonstrates the basic file import with IsoTools.
It uses a small test data set contained in this repository, and should run within seconds. The paths are relative to the root of the repository.
For more comprehensive real world examples see the [tutorials](https://isotools.readthedocs.io/en/latest/tutorials.html "readthedocs").

```python
from isotools import Transcriptome
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
# import the reference annotation
transcriptome = Transcriptome.from_reference('tests/data/example.gff.gz')
# import the transcriptome data
for sa in ('CTL', 'VPA'):
    transcriptome.add_sample_from_bam(f'../tests/data/example_1_{sa}.bam', sample_name=sa, group=sa, platform='SequelII')
# save the imported file as pkl file (for faster import)
transcriptome.add_qc_metrics('../tests/data/example.fa')
transcriptome.save('../tests/data/example_1_isotools.pkl')
```

## Citation and feedback

* If you run into any issues, please use the [github issues report feature](https://github.com/HerwigLab/IsoTools2/issues).
* For general feedback, please write us an email to [yalan_bi@molgen.mpg.de](mailto:yalan_bi@molgen.mpg.de) and [herwig@molgen.mpg.de](mailto:herwig@molgen.mpg.de).
* If you use IsoTools in your publication, please cite the following paper in addition to this repository:
  * Lienhard, Matthias et al. “**IsoTools: a flexible workflow for long-read transcriptome sequencing analysis**.” Bioinformatics (Oxford, England) vol. 39,6 (2023): btad364. [doi:10.1093/bioinformatics/btad364](https://doi.org/10.1093/bioinformatics/btad364)
  * Bi, Yalan et al. “**IsoTools 2.0: Software for Comprehensive Analysis of Long-read Transcriptome Sequencing Data**.” Journal of molecular biology, 169049. 26 Feb. 2025, [doi:10.1016/j.jmb.2025.169049](https://doi.org/10.1016/j.jmb.2025.169049)
