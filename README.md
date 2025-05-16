# VACVPlaque
This repository contains code for the experiments shown in the paper "A digital photography dataset for Vaccinia Virus plaque quantification using Deep Learning" [Nat. Sci. Data paper link](https://www.nature.com/articles/s41597-025-05030-8)

## Citation 
Please cite as follows:

De, T., Thangamani, S., Urbański, A., Yakimovich, A.: A digital photography dataset for Vaccinia Virus plaque quantification using Deep Learning. Sci Data. 12, 719 (2025). https://doi.org/10.1038/s41597-025-05030-8.

```
@ARTICLE{De25-vacv,
  title     = "A digital photography dataset for Vaccinia Virus plaque
               quantification using Deep Learning",
  author    = "De, Trina and Thangamani, Subasini and Urba{\'n}ski, Adrian and
               Yakimovich, Artur",
  journal   = "Sci. Data",
  publisher = "Springer Science and Business Media LLC",
  volume    =  12,
  number    =  1,
  pages     = "719",
  month     =  apr,
  year      =  2025,
  copyright = "https://creativecommons.org/licenses/by/4.0"
}
```

## Abstract
Virological plaque assay is the major method of detecting and quantifying infectious viruses in research and diagnostic samples. Furthermore, viral plaque phenotypes contain information about the life cycle and spreading mechanism of the virus forming them. While some modernisations have been proposed, the conventional assay typically involves manual quantification of plaque phenotypes, which is both laborious and time-consuming. Here, we present an annotated dataset of digital photographs of plaque assay plates of Vaccinia virus - a prototypic propoxvirus. We demonstrate how analysis of these plates can be performed using deep learning by training models based on the leading architecture for biomedical instance segmentation - StarDist. Finally, we show that the entire analysis can be achieved in a single step by HydraStarDist - the modified architecture we propose.

## Installation
Please set up the environment using `conda` or `pip`. We recommend creating a new environment for this project. Navigate to the root directory of this project and run:

```
conda create -f stardist_environment.yml
conda activate stardist2
pip install -e .
```

For some systems, the following line may be needed:
```
CC=gcc-<GCC VERSION> CXX=g++-<GCC VERSION> pip install -e .
```

## Usage
Please use the scripts under ```examples/2D/models/stardist/```, ```examples/2D_hydra/models/stardist/``` and ```scripts/``` in combination with an appropriate config file from ```configs/``` to run the code. Please change the config with appropriate data, output and model weight paths.

Since HSD our branched architectures shares plenty of common functionality with SD, a slight switch is needed to ensure the correct methods are being referred to. Please comment out one of the two lines below from ```stardist/models/__init__.py``` to use either a non-branched or branched(hydra) architecture.

```
from .model2d import Config2D, StarDist2D, StarDistData2D
from .model2d_hydra import Config2D, StarDist2D, StarDistData2D
```
