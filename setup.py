from os import path

from setuptools import setup

repo_path = path.dirname(path.realpath(__file__))

with open(path.join(repo_path, "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()

setup(name="llm_playbooks", version="0.0.1", install_requires=requirements)
