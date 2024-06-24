# DEEIA
The code for Beyond Single-Event Extraction: Towards Efficient Document-Level Multi-Event Argument Extraction , Findings of ACL2024. The paper can be seen on arXiv (https://arxiv.org/abs/2405.01884).

The complete code is currently being organized and will be released later. At present, we have made available the core code from the paper, including the Dependency-guided Encoder (DE) module and the Event-specific Information Aggregation (EIA) module.

## Dependency-guided Encoder (DE) Module

The DE module's operations are integrated into the transformer architecture. For details, please refer to the `modeling_bart.py` and `modeling_roberta_.py` files.

## Event-specific Information Aggregation (EIA) Module

The definition of inter / intra-event dependencies can be found in the data processing file `processor_multiarg.py`.

Detailed comments are provided within the files for your reference. Our code is built upon the work of PAIE (https://github.com/mayubo2333/PAIE), and we hope it will be helpful to you!

