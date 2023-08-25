from os import path

from setuptools import setup

dir_path = path.dirname(path.realpath(__file__))

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(name="llm_playbooks", version="0.0.1", install_requires=requirements)
