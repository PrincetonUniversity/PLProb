#!/bin/usr/env python
import os
import sys
import setuptools
from pathlib import Path


setuptools.setup(
    name='plprob',
    version='0.1',
    description="Peak load probabilistic prediction platform",

    author='Xinshuo Yang',
    author_email='xy3134@princeton.edu',

    packages=['plprob'],
    package_data={'plprob': ['../data/*/*.csv']},

    install_requires=['numpy', 'matplotlib', 'pandas', 'scipy',
                      'dill', 'statsmodels', 'cffi', 'jupyterlab',
                      'seaborn', 'openpyxl', 'rpy2', 'geopandas',
                      'scikit-learn', 'ipywidgets', 'astral'],
    )


os.system("curl -L https://carmona.princeton.edu/SVbook/Rsafd.zip "
          "--output Rsafd.zip")
os.system("unzip Rsafd.zip")
os.system('R -e "install.packages(\'Rsafd\', repos = NULL, type=\'source\')"')
os.system("rm -rf Rsafd Rsafd.zip")

# hacky way tp get around flawed TclTk installs on MacOS
tcltk_path = Path(sys.exec_prefix,
                  "lib", "R", "library", "tcltk", "libs", "tcltk.so")
if not tcltk_path.exists():
    os.system("cp {} {}".format(tcltk_path.with_suffix(".dylib"), tcltk_path))