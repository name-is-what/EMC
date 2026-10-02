# [NeurIPS 2026] Efficient Memory Crystallization for Graph Learning under Non-Stationary Distribution Shifts
This repository is the official implementation of "Efficient Memory Crystallization for Graph Learning under Non-Stationary Distribution Shifts" accepted by the 40th Conference on Neural Information Processing Systems (NeurIPS 2026).


[![Black Logo](framework.png)](https://openreview.net/forum?id=CQBiYvXLgz)

------

## 0. Abstract


Deep graph learning models deployed in real-world systems often need to cope with non-stationary environments, where the underlying graph distribution drifts continually over time. Prevailing solutions rely on training auxiliary generative modules to synthesize memory graphs for cross-domain adaptation, which incurs substantial computational overhead and scales poorly under prolonged distribution shifts. We argue that a more economical path exists: rather than generating memory, one can crystallize it. To this end, we propose Efficient Memory Crystallization (EMC), a training-free test-time framework that distills each incoming graph domain into a compact, semantically faithful memory through a closed-form solution to a memory-oriented distribution-matching objective, thereby eliminating redundant domain information under continual covariate shifts. To preserve both generalizability and adaptability as the model traverses a long sequence of target domains, EMC further models inter-domain dependencies through state-evolving memories and admits a theoretically grounded, tighter generalization error bound than direct adaptation. Extensive experiments demonstrate the superior performance of EMC over state-of-the-art baselines on graphs under non-stationary distribution shifts, while reducing average runtime by 87.4% and GPU memory consumption by 92.4% relative to the recent competitor, making continual graph adaptation practical at scale.


## 1. Requirements

Main package requirements:

- `CUDA == 11.1`
- `Python == 3.7.12`
- `PyTorch == 1.8.0`
- `PyTorch-Geometric == 2.0.0`

To install the complete requiring packages, use the following command at the root directory of the repository:

```setup
pip install -r requirements.txt
```


## 2. Quick Start
Just run the script corresponding to the experiment and dataset you want. For instance:





## 3. Citation
If you find this repository helpful, please consider citing the following paper. We welcome any discussions with [hou_yue@buaa.edu.cn](mailto:hou_yue@buaa.edu.cn).

```bibtex

```
