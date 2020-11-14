from multiprocessing import Process
import os
import datetime
from time import sleep
import psutil
import numpy as np
from subprocess import Popen, PIPE
from distutils import spawn
import os
import math
import random
import time
import sys
import platform


__version__ = '1.4.0'

def info(title, count=1000):
    print(title)
    i = 0
    l = []
    pid = os.getpid()
    print('info module name:', __name__)
    print('info parent process:', os.getppid())
    print('info process id:', pid)
    while i < count:
      now = datetime.datetime.now()
      timestamp = '{}-{}-{}-{}-{}-{}'.format(
                            now.year, now.month, now.day,
                            now.hour, now.minute, now.second)
      print(f'info pid={pid} time={timestamp}')
      sleep(0.2)
      for _ in range(5):
        a = np.random.rand(100, 200)
        b = np.random.rand(200, 300)
        l.append(np.dot(a, b))
      i+=1


def log(main_pid):
    # info('function f')
    p = psutil.Process(main_pid)
    pid = os.getpid()
    gpu = getGPUs()[0]
    print('hello', main_pid)
    print('log parent process:', os.getppid())
    print('log process id:', pid)
    print('log cpu:', p.cpu_percent())
    print('log mem:', p.memory_percent())
    print('log GPU:', gpu.load)
    print('log GPU mem:', gpu.memoryUtil)
    i = 0
    while True:
      now = datetime.datetime.now()
      timestamp = '{}-{}-{}-{}-{}-{}'.format(
                            now.year, now.month, now.day,
                            now.hour, now.minute, now.second)
      print(f'{i}, {timestamp}, log-pid={pid}, monitor-pid={p.pid}, cpu={p.cpu_percent()}, mem={p.memory_percent()}')
      sleep(1)
      i+=1

if __name__ == '__main__':
    main_pid = os.getpid()
    p = Process(target=log, args=(main_pid,))
    p.start()
    sleep(5)
    info('main line')
    p.join()