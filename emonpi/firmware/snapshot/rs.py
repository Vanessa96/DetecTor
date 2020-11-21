#!/usr/bin/python
import serial

from datetime import datetime

ser = serial.Serial('/dev/ttyAMA0', 38400)

ser.write(b'f10')
# ser.write(b'f20')
# ser.write(b'c1')
try:
  while True:
    response = ser.readline()
  #  z = response.split(" ")
    try:
      print("{},{}".format(datetime.now(), response.decode().strip()))
    except UnicodeDecodeError as e:
      print(e, 'decode error, skip')
      continue
    # print("{}\n".format(datetime.now()))
    print()

except KeyboardInterrupt:
  ser.close()

