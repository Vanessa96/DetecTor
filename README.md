# Measure Energy of Running NLP Models

## Setup

For software methodology

### Install

[pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)
cuda 10.2 cudnn 8.0.3

`pyenv install 3.7.9`

`pyenv virtualenv 3.7.9 nrg`

`pyenv activate nrg`

`pip install -r requirements.txt`

`pyenv deactivate`

setup OpenEnergyMonitor, see `emonpi/readme.md`
compile resources profiler, `cd rpof && make`

### Usage

for every experiment, start the energy monitor first (`python energy_monitor.py -o energy.csv`)

then `cd rpof; ./rprof 170 res.csv 50` means profile every 170 ms for 50s

last, start run the model exp script like in `cmd.sh`
