# CSCE-642: Deep Reinforcement Learning

## Setup

SWIG is required for installing Box2D. It can be installed on Linux by running 
```bash
sudo apt-get install swig build-essential python-dev python3-dev
```
and on Mac by running
```bash
brew install swig
```

For setting up the environment, we recommend using conda.

For linux run
```bash
conda env create --file=environment_linux.yml
```


For mac run (Tested on M1 and M2 chips)
```bash
conda env create --file=environment_mac.yml
```

For Windows consider setting up a Python environment with version 3.9.16. Install the packages given by
```bash
pip install -r requirements.txt
```
