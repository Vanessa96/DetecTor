#!/usr/bin/python3
import argparse
import serial

from datetime import datetime
import time


def main(args):
  output_file = args.output_file
  of = open(output_file, 'w')

  ser = serial.Serial('/dev/ttyAMA0', 38400)

  # ser.write(b'f10')
  # ser.write(b'f20')
  # ser.write(b'c1')
  try:
    while True:
      response = ser.readline()
    #  z = response.split(" ")
      try:
        content = response.decode().strip()
        save_data = f'{time.clock_gettime(time.CLOCK_REALTIME):.6f},{content}\n'
        of.write(save_data)
        print(f'{datetime.now()},{save_data}')
      except UnicodeDecodeError as e:
        print(e, 'decode error, skip')
        continue
      # print("{}\n".format(datetime.now()))
      print()

  except KeyboardInterrupt:
    ser.close()
    of.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-o", "--output_file", type=str, required=True,
                      help="output file")
  main(parser.parse_args())
