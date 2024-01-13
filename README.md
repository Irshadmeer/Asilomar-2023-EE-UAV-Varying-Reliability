# Reinforcement Learning Based Dynamic Power Control for UAV Mobility Management

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Irshadmeer/Asilomar-2023-EE-UAV-Varying-Reliability/HEAD)
![GitHub](https://img.shields.io/github/license/Irshadmeer/Asilomar-2023-EE-UAV-Varying-Reliability)
[![arXiv](https://img.shields.io/badge/arXiv-2312.04742-informational)](https://arxiv.org/abs/2312.04742)


This repository is accompanying the paper "Reinforcement Learning Based Dynamic
Power Control for UAV Mobility Management" (Irshad Meer, Karl-L. Besser,
Mustafa Ozger, Vincent Poor, and Cicek Cavdar, 2023 Asilomar Conference on
Signals, Systems, and Computers, Oct. 2023.
[arXiv:2312.04742](https://arxiv.org/abs/2312.04742)).


## File List
The following files are provided in this repository:

- `baseline.py`: Python module that contains the comparison/baseline algorithms
- `data_logger.py`: Python module that contains a custom callback for saving data.
- `environment.py`: Python module that contains the `gym` environment.
- `loggers.py`: Python module that contains a custom callback for saving data.
- `main_training.py`: Python script that runs the training.
- `movement.py`: Python module that contains the implementation of the stochastic UAV movement model.
- `reliability.py`: Python module that contains functions for calculating the outage probability.
- `test.py`: Python script that runs the testing of the trained model.
- `util.py`: Python module that contains utility functions.


## Usage
### Running it online
You can use services like [CodeOcean](https://codeocean.com) or
[Binder](https://mybinder.org/v2/gh/Irshadmeer/Asilomar-2023-EE-UAV-Varying-Reliability/HEAD) to run
the scripts online.

### Local Installation
If you want to run it locally on your machine, make sure that Python3 and all
required libraries are installed.


## Acknowledgements
This research was supported in part by the CELTIC-NEXT Project, 6G for
Connected Sky (6G-SKY), with funding received from Vinnova, Swedish Innovation
Agency, by the German Research Foundation (DFG) under grant BE 8098/1-1, and by
the U.S National Science Foundation under Grants CNS-2128448 and ECCS-2335876.



## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```bibtex
@article{Meer2023reinforcement,
  author = {Meer, Irshad A. and Besser, Karl-Ludwig and Ozger, Mustafa and Poor, H. Vincent and Cavdar, Cicek},
  title = {Reinforcement Learning Based Dynamic Power Control for UAV Mobility Management},
  booktitle = {2023 57th Asilomar Conference on Signals, Systems, and Computers},
  year = {2023},
  month = {10},
  publisher = {IEEE},
  venue = {Pacific Grove, CA, USA},
  archiveprefix = {arXiv},
  eprint = {2312.04742},
  primaryclass = {cs.IT},
}
```
