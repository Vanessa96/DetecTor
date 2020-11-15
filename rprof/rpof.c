#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

typedef unsigned long ctime_t;
ctime_t gettime(void) {
   struct timespec ts;
   clock_gettime(CLOCK_REALTIME, &ts);
   return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3; // microsecond
}

// ---- SIGNAL HANDLING -------------------------------------------------------

static int interrupt = 0;

static void
signal_callback_handler(int signum)
{
  if (signum == SIGINT || signum == SIGTERM)
    interrupt = SIGTERM;
} /* signal_callback_handler */

// ----------------------------------------------------------------------------

static int
strtonum(char * str, ctime_t * num) {
  char * p_err;

  *num = strtol(str, &p_err, 10);

  return  (p_err == str || *p_err != 0) ? 1 : 0;
}

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
  
  ctime_t profile_interval = 100;
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

  ctime_t timeout = 10; // 10 s
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
  ctime_t start_time = gettime();
  ctime_t sample_time = gettime();
  while (interrupt == 0 && (sample_time - start_time) < timeout)
  {

    sample_time = gettime();
    printf("sample_time=%lu\n", sample_time); 
    fprintf(output_file, "%lu, %s %f %d\n", sample_time, "are", 1.2, 2012);
    ctime_t delta = gettime() - sample_time;
    if (delta < profile_interval){
      usleep(profile_interval - delta);
    }
  }

  fclose(output_file);
  printf("elapsed %.3f ms\n", (sample_time - start_time)/1e3);

  return retval;
}
