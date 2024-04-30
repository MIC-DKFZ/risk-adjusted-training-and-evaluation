'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

from setuptools import setup, find_packages


setup(
    name="mod_metrics",
    version="1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'mod_metrics_v1=mod_metrics.__main__:main'
        ]
    }
)