# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


import os.path

readme = ""
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.rst")
if os.path.exists(readme_path):
    with open(readme_path, "rb") as stream:
        readme = stream.read().decode("utf8")


setup(
    long_description=readme,
    name="calimu",
    version="0.1.0",
    description="A tool for calibrating IMUs",
    python_requires="==3.*,>=3.7.0",
    project_urls={"repository": "https://github.com/simleek/displayarray"},
    author="SimLeek",
    author_email="simulator.leek@gmail.com",
    license="MIT",
    entry_points={"console_scripts": ["calimu = calimu.gui:main"]},
    packages=[
        "calimu",
        "calimu.imu",
        "calimu.imu.devices",
        "calimu.pcl_algo",
    ],
    package_dir={"": "."},
    package_data={},
    install_requires=[
        "numpy>=1.23.1",
        "pyserial>=3.5",
        "scikit_learn>=1.1.2",
        "svtk>=0.2.0",
        "vtk>=9.1.0",
    ],
    extras_require={
        "dev": [
            "black==18.*,>=18.3.0.a0",
            "coverage==4.*,>=4.5.0",
            "mypy==0.*,>=0.740.0",
            "pydocstyle==4.*,>=4.0.0",
            "pytest==5.2.1",
            "sphinx==2.*,>=2.2.0",
            "tox==3.*,>=3.14.0",
            "tox-gh-actions==0.*,>=0.3.0",
            "typing==3.7.4.1",
            "wheel==0.*,>=0.30.0",
        ],
    },
)
