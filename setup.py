from setuptools import setup, find_packages

install_requires = [
    "allennlp>=0.9.0",
    "datasets @ git+https://gitlab.com/kb-completion/datasets.git",
    "wandb==0.8.15",
]

setup(
    name='models',
    version='0.0.1',
    description='AllenNLP style models for KB Completion',
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_data={'models': ['py.typed']},
    install_requires=install_requires,
    zip_safe=False)
