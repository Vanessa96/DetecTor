# OpenEnergyMonitor emonPi

overview: https://guide.openenergymonitor.org/setup/
we use emonPi

## emonPi Setup
follow emonPi hardware and software setup

https://guide.openenergymonitor.org/setup/install/

## Firmware Modification

follow guide from https://guide.openenergymonitor.org/technical/compiling/

install PlatformIO

clone https://github.com/openenergymonitor/emonpi
replace following with corresponding files in this repo,
firmware/src/rf.ino # just added two extra commands, so we can change which current sensor to get the energy source, and how many no_of_half_wavelengths we need
firmware/src/src.ino # only added 3 variables to log ct (power source, either ct1 or ct2 to log) and power values for ct1 and ct2, removed unused logics in the loop function.
firmware/compiled/update # just a simple script to flash the firmware.

`sudo pio run && compiled/update .pio/build/emonpi/firmware.hex` to compile and flash the firmware.

put firmware/rs.py # sampling script, run on the raspberry pi.

also see `https://github.com/csarron/emonpi` if for any reason the above change does not work.

### Sampling energy using emonPi

`pip install pyserial==3.4`

https://github.com/csarron/emonpi/blob/master/firmware/energy_monitor.py
