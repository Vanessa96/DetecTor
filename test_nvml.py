import datetime
from time import sleep
from py3nvml.py3nvml import *
nvmlInit()
print("Driver Version: {}".format(nvmlSystemGetDriverVersion()))
# e.g. will print:
#   Driver Version: 352.00
deviceCount = nvmlDeviceGetCount()
handles = []
names = []
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    name = nvmlDeviceGetName(handle)
    # nvmlDeviceSetAccountingMode(handle, True)
    # nvmlDeviceSetPersistenceMode(handle, True)
    print("Device {}: {}".format(i, name))
    handles.append(handle)
    names.append(name)
n = 0
handles.pop()
"""
NVML_TOTAL_POWER_SAMPLES = 0
To represent total power drawn by GPU.
NVML_GPU_UTILIZATION_SAMPLES = 1
To represent percent of time during which one or more kernels was executing on the GPU.
NVML_MEMORY_UTILIZATION_SAMPLES = 2
To represent percent of time during which global (device) memory was being read or written.
NVML_ENC_UTILIZATION_SAMPLES = 3
To represent percent of time during which NVENC remains busy.
NVML_DEC_UTILIZATION_SAMPLES = 4
To represent percent of time during which NVDEC remains busy.
NVML_PROCESSOR_CLK_SAMPLES = 5
To represent processor clock samples.
NVML_MEMORY_CLK_SAMPLES = 6
To represent memory clock samples.
"""
sample_types = [0, 1, 2, 5, 6]
powers = [[] for _ in range(len(handles))]
now = datetime.datetime.now()
last_timestamp = int(now.timestamp()*1e6)
sleep(1)
while n < 10:
  now = datetime.datetime.now()
  for i, h in enumerate(handles):
    p1 = nvmlDeviceGetPowerUsage(h)
    # pids = nvmlDeviceGetComputeRunningProcesses(h) #nvmlDeviceGetAccountingPids(h)
    # print(';'.join([f'{p.pid}={p.usedGpuMemory}' for p in pids]))
    st = nvmlDeviceGetUtilizationRates(h)
    # st = [nvmlDeviceGetAccountingStats(h, p.pid) for p in pids]
    # print(i, )
    samples = [(nvmlSamplingType_t(sample_type).name, [(datetime.datetime.fromtimestamp(s.timeStamp/1e6).isoformat(), s.sampleValue.uiVal) for s in nvmlDeviceGetSamples(h, sample_type, last_timestamp)[1]]) for sample_type in sample_types]
    # _, samples = nvmlDeviceGetSamples(h, 3, 0)
    # samples = [f'{datetime.datetime.fromtimestamp(s.timeStamp/1e6).isoformat()}={s.sampleValue.uiVal}' for s in samples]
    p = (now.isoformat(), p1, str(st), len(samples), samples)
    powers[i].append(p)
    print(i, p[0], p[1], p[2], p[3])
    for (name, pt) in p[4]:
      print('\t', name, len(pt))
      for ts, tv in pt:
        print('\t\t', ts, tv)
  last_timestamp = int(now.timestamp()*1e6)
  sleep(1)
  n+=1

print("======all samples======")
for i, power in enumerate(powers):
  for p in power:
    print(i, p[0], p[1], p[2], p[3])
    for (n, pt) in p[4]:
      print('\t', n, len(pt))
      for ts, tv in pt:
        print('\t\t', ts, tv)