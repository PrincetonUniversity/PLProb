#!/bin/usr/env python
import os
import sys
import setuptools
from pathlib import Path


setuptools.setup(
    name='plprob',
    version='0.11',
    description="Peak load probabilistic prediction platform",

    author='Xinshuo Yang, Amit Solomon',
    author_email='xy3134@princeton.edu, as3993@princeton.edu',

    packages=['plprob'],
    package_data={'plprob': ['../data/*/*.csv']},

    install_requires=['numpy', 'matplotlib', 'pandas', 'scipy==1.11.4',
                      'dill', 'statsmodels', 'cffi', 'jupyterlab',
                      'seaborn', 'openpyxl', 'geopandas',
                      'scikit-learn', 'ipywidgets', 'astral', 'zstandard'],
    )


# os.system("curl -L https://carmona.princeton.edu/SVbook/Rsafd.zip "
#           "--output Rsafd.zip")
# os.system("unzip Rsafd.zip")
# os.system('R -e "install.packages(\'Rsafd\', repos = NULL, type=\'source\')"')
# os.system("rm -rf Rsafd Rsafd.zip")
# os.system('R -e "install.packages(\'Rsafd_From_Scratch_9_28_2023\', repos = NULL, type=\'source\')"')

#TODO: Not sure if this is needed
# # hacky way tp get around flawed TclTk installs on MacOS
# tcltk_path = Path(sys.exec_prefix,
#                   "lib", "R", "library", "tcltk", "libs", "tcltk.so")
# if not tcltk_path.exists():
#     os.system("cp {} {}".format(tcltk_path.with_suffix(".dylib"), tcltk_path))