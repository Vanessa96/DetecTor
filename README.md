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

## Usage

### WattsUp Meter Logger

This is hardware based methodology, it's more accurate and reliable.
- `pip install pyserial numpy matplotlib PyQt5`
- plug in the usb to your computer, then `sudo chown $USER /dev/ttyUSB0`

- log energy: `python wattsup.py -l -o sample.log`

- plot energy: `python plot.py sample.log sample.png`
