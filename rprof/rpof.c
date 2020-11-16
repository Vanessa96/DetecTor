#include "cpu.h"
#include "mem.h"
#include "nvml.h"

// ---- SIGNAL HANDLING -------------------------------------------------------

static int interrupt = 0;

static void
signal_callback_handler(int signum)
{
  if (signum == SIGINT || signum == SIGTERM)
    interrupt = SIGTERM;
} /* signal_callback_handler */

// ----------------------------------------------------------------------------

static const char * usage_msg = \
"Usage: rprof [OPTION...] command [ARG...]\n"
"Rprof -- A high frequency resources (cpu, mem, gpu, gpu mem) profiler.\n"
"\n"
"  profile_interval (ms)    Sampling profile_interval (default is 100 ms).\n"
"  output_file              Specify an output file for the collected samples.\n"
"  -b, --batch              Size of batch writing data points.\n"
"  timeout (s)             Approximate start up wait time. Increase on slow\n"
"                             machines (default is 10s).\n";

int main(int argc, char ** argv) {
  int retval = 0;
  if (argc <= 1) {
    puts(usage_msg);
    return retval;
  }
  int a      = 1;
  
  utime_t profile_interval = 100;
  if (argc >= 2){
    if (1 == strtonum(argv[1], &profile_interval)){
      printf("failed to get profile_interval, %s, use default 100 ms\n", argv[1]); 
      // return 1;
    };
  }
  printf("profile_interval=%lu ms\n", profile_interval);
  profile_interval = profile_interval * 1e3;
  FILE* output_file = stdout;
  if (argc >= 3){
    char* output_filename = argv[2];
    output_file = fopen(output_filename, "w");
    if (NULL == output_file)
    {
      printf("failed to write, %s, write to stdout\n", output_filename);
      output_file = stdout;
    }
    else {
      printf("log to %s\n", output_filename);
    }
  }

  utime_t timeout = 10; // 10 s
  if (argc >= 4){
    if (1 == strtonum(argv[3], &timeout)){
      printf("failed to get timeout, %s, default to 10 seconds\n", argv[1]); 
      // return 1;
    };
  }
  printf("timeout=%lu s\n", timeout);
  timeout = timeout*1e6;
  // Register signal handler for Ctrl+C and terminate signals.
  signal(SIGINT, signal_callback_handler);
  signal(SIGTERM, signal_callback_handler);

  // Starting profiling
  utime_t start_time = gettime();
  utime_t sample_time = gettime();

  nvmlReturn_t result;
  unsigned int device_count;
  char driver_version[80];
  result = nvmlInit();
  result = nvmlSystemGetDriverVersion(driver_version, 80);
  printf("\n Driver version:  %s \n\n", driver_version);
  result = nvmlDeviceGetCount(&device_count);
  printf("Found %d device%s\n\n", device_count, device_count!= 1 ? "s" : "");
  printf("Listing devices:\n");
  for (unsigned i = 0; i < device_count; i++) 
  {
    nvmlDevice_t device;  
    char name[64];  
    nvmlComputeMode_t compute_mode;
    result = nvmlDeviceGetHandleByIndex(i, &device);
    result = nvmlDeviceGetName(device, name, sizeof(name)/sizeof(name[0]));
    printf("%d. %s \n", i, name);
  } 
  
  unsigned long print_count = 0;
  struct cpustat cpu_util_prev, cpu_util_cur;
  struct meminfo mem_util;
  get_stats(&cpu_util_prev, -1);
  usleep(profile_interval); // sleep one interval to avoid negative first sample
  while (interrupt == 0 && (sample_time - start_time) < timeout)
  {

    get_stats(&cpu_util_cur, -1);
    double cpu_util = calculate_load(&cpu_util_prev, &cpu_util_cur);
    double mem_usage = calculate_mem_usage(&mem_util);
    sample_time = gettime();
    fprintf(output_file, "%lu, %.1f %.1f\n", sample_time, cpu_util, mem_usage);
    if (print_count%10==0)
    {
      printf("\33[2K\r");
      printf("sample: %lu, %.1f %.1f\n", sample_time, cpu_util, mem_usage);
    }
    print_count++;

    get_stats(&cpu_util_prev, -1);
    utime_t delta = gettime() - sample_time;
    if (delta < profile_interval){
      usleep(profile_interval - delta);
    }
  }

  fclose(output_file);
  result = nvmlShutdown();
  printf("\nelapsed %.3f ms\n", (sample_time - start_time)/1e3);
  return retval;
}
