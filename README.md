# PLProb #

This package computes probability of peak lead and predict coincident peak (CP) load for ISOs/RTOs
using historical load actual and forecast values.



## Installation ##

PLProb is available for installation on Linux-based operating systems such as Ubuntu as well as macOS. To install
PLProb, first clone this repository at the latest release:

```gh repo clone PrincetonUniversity/PLProb```

Next, navigate to the cloned directory to create and activate the conda environment containing the prerequisite
packages for PLProb:

```conda env create -f environment.yml```

```conda activate plprob```

From within the same directory, complete installation of PLProb by running:

```pip install .```


## Running PLProb for PJM, NYISO or ISONE ##
Please see the Jupyter notebooks available in the `example/` directory for an overview of how PLProb works.
