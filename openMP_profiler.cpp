#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <iostream>
#include "omp.h"
#include <string.h>
#include <stdlib.h>
#include "sys/types.h"
#include "sys/sysinfo.h"
#include "sys/times.h"
#include "sys/vtimes.h"
#include <cstdio>
#include <vector>
#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"
#include <fstream>
#include <iomanip>
#include <float.h>
#include <pthread.h>
#include <limits>
#include <unistd.h>
#include <sigar.h>
# include <cstdlib>
# include <iomanip>
# include <ctime>
# include <cmath>
# include <omp.h>

extern "C" 
{
#include <sigar_format.h>
}

using namespace std;
struct sysinfo memInfo;
clock_t lastCPU, lastSysCPU, lastUserCPU;
int numProcessors;
static unsigned long long lastTotalUser, lastTotalUserLow, lastTotalSys, lastTotalIdle;

pthread_mutex_t lock;


struct benchmark_param{
	int n;
	int num_tries;
	int num_thr_start;
	int num_thr_end;
};

/* One record per each round/run of a benchmark
 * Each round/run consists of a specific number of runs
 * given with the variable num_tries
 */

/* Captures the state of the runtime during one run
 * Note : Since all the benchmarks are ran concurrently using pthreads,
 *		  the state of the runtime has to be reperesnetative of the 
 *	      aggregate state, not just the specific thread
 */
struct dataset_record{
	int num_threads; // the optimal number of threads for this specific run
	double cpu_usage;
	double pid_cpu_usage;
	double ram_usage;
	double pid_page_faults;// percent out of total
	double pid_ram_usage;
	double vm_usage;
	double pid_vm_read_usage;
	double pid_vm_write_usage;
	long processes; //the number of processes and threads created, which includes (but is not limited to) those created by calls to the fork() and clone() system calls
	double procs_running;
	double procs_blocked;
	//long ctxt;  // the total number of context switches across all CPUs
    // Note: for the current cpu
    double steal_time; // It counts the ticks spent executing other virtual hosts
	double user;
	double sys;
	double nice;
	double idle;
	double wait;
	double soft_irq;
	double irq; 
	double total;
	// for all available cpus
	double lst_cpu_usage;
	unsigned long long lst_stolen; // It counts the ticks spent executing other virtual hosts
	unsigned long long lst_user;
	unsigned long long lst_sys;
	unsigned long long lst_nice;
	unsigned long long lst_idle;
	unsigned long long lst_wait;
	unsigned long long lst_soft_irq;
	unsigned long long lst_irq; 
	unsigned long long lst_total;
};
bool forest_created = false;

# define NV 1440
# define dataset_path "experiments/alg/exp11.data"
# define forest_path "experiments/alg/exp11.forest"
# define prediction_log "experiments/alg/exp29_ftt.txt"

int run_dijkstra(int num_thrs);
int *dijkstra_distance ( int ohd[NV][NV], int num_thrs);
void find_nearest ( int s, int e, int mind[NV], bool connected[NV], int *d, 
  int *v );
void init ( int ohd[NV][NV] );
void timestamp ( );
void update_mind ( int s, int e, int mv, bool connected[NV], int ohd[NV][NV], 
  int mind[NV] );

int run_dijkstra(int num_thrs){
  int i;
  int i4_huge = 2147483647;
  int j;
  int *mind;
  int ohd[NV][NV];

  timestamp ( );
//
//  Initialize the problem data.
//
  init ( ohd );
//
//  Carry out the algorithm.
//
  mind = dijkstra_distance(ohd, num_thrs);
//
//  Free memory.
//
  delete [] mind;
//
//  Terminate.
//
  //cout << "\n";
  //cout << "DIJKSTRA_OPENMP\n";
  //cout << "  Normal end of execution.\n";

  //cout << "\n";
  timestamp ( );
  return 0;
}

int *dijkstra_distance ( int ohd[NV][NV], int num_thrs)
{
  bool *connected;
  int i;
  int i4_huge = 2147483647;
  int md;
  int *mind;
  int mv;
  int my_first;
  int my_id;
  int my_last;
  int my_md;
  int my_mv;
  int my_step;
  int nth;
//
//  Start out with only node 0 connected to the tree.
//
  connected = new bool[NV];
  connected[0] = true;
  for ( i = 1; i < NV; i++ )
  {
    connected[i] = false;
  }
//
//  Initialize the minimum distance to the one-step distance.
//
  mind = new int[NV];

  for ( i = 0; i < NV; i++ )
  {
    mind[i] = ohd[0][i];
  }
//
//  Begin the parallel region.
//
 
  # pragma omp parallel num_threads(num_thrs) private ( my_first, my_id, my_last, my_md, my_mv, my_step ) \
  shared ( connected, md, mind, mv, nth, ohd )
  {
    my_id = omp_get_thread_num ( );
    nth = omp_get_num_threads ( ); 
    my_first =   (   my_id       * NV ) / nth;
    my_last  =   ( ( my_id + 1 ) * NV ) / nth - 1;    
    
//
//  Attach one more node on each iteration.
//
    for ( my_step = 1; my_step < NV; my_step++ )
    {
//
//  Before we compare the results of each thread, set the shared variable 
//  MD to a big value.  Only one thread needs to do this.
//
      # pragma omp single 
      {
        md = i4_huge;
        mv = -1; 
      }
//
//  Each thread finds the nearest unconnected node in its part of the graph.
//  Some threads might have no unconnected nodes left.
//
      find_nearest ( my_first, my_last, mind, connected, &my_md, &my_mv );
//
//  In order to determine the minimum of all the MY_MD's, we must insist
//  that only one thread at a time execute this block!
//
      # pragma omp critical
      {
        if ( my_md < md )  
        {
          md = my_md;
          mv = my_mv;
        }
      }
//
//  This barrier means that ALL threads have executed the critical
//  block, and therefore MD and MV have the correct value.  Only then
//  can we proceed.
//
      # pragma omp barrier
//
//  If MV is -1, then NO thread found an unconnected node, so we're done early. 
//  OpenMP does not like to BREAK out of a parallel region, so we'll just have 
//  to let the iteration run to the end, while we avoid doing any more updates.
//
//  Otherwise, we connect the nearest node.
//
      # pragma omp single 
      {
        if ( mv != - 1 )
        {
          connected[mv] = true;
         // cout << "  P" << my_id
         //      << ": Connecting node " << mv << "\n";;
        }
      }
//
//  Again, we don't want any thread to proceed until the value of
//  CONNECTED is updated.
//
      # pragma omp barrier
//
//  Now each thread should update its portion of the MIND vector,
//  by checking to see whether the trip from 0 to MV plus the step
//  from MV to a node is closer than the current record.
//
      if ( mv != -1 )
      {
        update_mind ( my_first, my_last, mv, connected, ohd, mind );
      }
//
//  Before starting the next step of the iteration, we need all threads 
//  to complete the updating, so we set a BARRIER here.
//
      #pragma omp barrier
    }

  }

  delete [] connected;

  return mind;
}

void find_nearest ( int s, int e, int mind[NV], bool connected[NV], int *d, 
  int *v )
{
  int i;
  int i4_huge = 2147483647;

  *d = i4_huge;
  *v = -1;
  for ( i = s; i <= e; i++ )
  {
    if ( !connected[i] && mind[i] < *d )
    {
      *d = mind[i];
      *v = i;
    }
  }
  return;
}

void init ( int ohd[NV][NV] )
{
  int i;
  int i4_huge = 2147483647;
  int j;
  srand(time(NULL));
  for ( i = 0; i < NV; i++ )  
  {
    for ( j = 0; j < NV; j++ )
    {
      if ( i == j )
      {
        ohd[i][i] = 0;
      }
      else
      {
		int flip = rand()%3+1;
		if(flip % 2 == 0){
		ohd[i][j] = i4_huge;
		}
		else{
        ohd[i][j] = rand()%500 + 1;
		}
        
      }
    }
  }

  return;
}
void timestamp ( )
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct std::tm *tm_ptr;
  size_t len;
  std::time_t now;

  now = std::time ( NULL );
  tm_ptr = std::localtime ( &now );

  len = std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );

 // std::cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}

void update_mind ( int s, int e, int mv, bool connected[NV], int ohd[NV][NV], 
  int mind[NV] )
{
  int i;
  int i4_huge = 2147483647;

  for ( i = s; i <= e; i++ )
  {
    if ( !connected[i] )
    {
      if ( ohd[mv][i] < i4_huge )
      {
        if ( mind[mv] + ohd[mv][i] < mind[i] )  
        {
          mind[i] = mind[mv] + ohd[mv][i];
        }
      }
    }
  }
  return;
}

/*
 * Funcitons for the the multitask benchmark
 * */
 
int run_multitask(int num_thrs);
int *prime_table ( int prime_num );
double *sine_table ( int sine_num );

int run_multitask (int num_thrs)
{
  int prime_num;
  int *primes;
  int sine_num;
  double *sines;
  double wtime;
  double wtime1;
  double wtime2;

  timestamp ( );
  
  prime_num = 12500;
  sine_num = 12500;

  wtime = omp_get_wtime ( );

# pragma omp parallel num_threads(num_thrs) shared ( prime_num, primes, sine_num, sines ) 
{
  # pragma omp sections
  {
    # pragma omp section
    {
      wtime1 = omp_get_wtime ( );
      primes = prime_table ( prime_num );
      wtime1 = omp_get_wtime ( ) - wtime1;
    }
    # pragma omp section
    {
      wtime2 = omp_get_wtime ( );
      sines = sine_table ( sine_num );
      wtime2 = omp_get_wtime ( ) - wtime2;
    }
  }
}
  wtime = omp_get_wtime ( ) - wtime;

  free ( primes );
  free ( sines );
//
//  Terminate.
//
  timestamp ( );

  return 0;
}

int *prime_table ( int prime_num )
{
  int i;
  int j;
  int p;
  int prime;
  int *primes;

  primes = ( int * ) malloc ( prime_num * sizeof ( int ) );

  i = 2;
  p = 0;

  while ( p < prime_num )
  {
    prime = 1;

    for ( j = 2; j < i; j++ )
    {
      if ( ( i % j ) == 0 )
      {
        prime = 0;
        break;
      }
    }
      
    if ( prime )
    {
      primes[p] = i;
      p = p + 1;
    }
    i = i + 1;
  }

  return primes;
}

double *sine_table ( int sine_num )
{
  double a;
  int i;
  int j;
  double pi = 3.141592653589793;
  double *sines;

  sines = ( double * ) malloc ( sine_num * sizeof ( double ) );

  for ( i = 0; i < sine_num; i++ )
  {
    sines[i] = 0.0;
    for ( j = 0; j <= i; j++ )
    {
      a = ( double ) ( j ) * pi / ( double ) ( sine_num - 1 );
      sines[i] = sines[i] + sin ( a );
    }
  }

  return sines;
}

// Functions for the poisson benchmark

# define NX 151
# define NY 151

int run_poisson ( int num_thrs);
double r8mat_rms ( int m, int n, double a[NX][NY] );
void rhs ( int nx, int ny, double f[NX][NY] );
void sweep ( int nx, int ny, double dx, double dy, double f[NX][NY], 
  int itold, int itnew, double u[NX][NY], double unew[NX][NY], int num_thrs );
double u_exact ( double x, double y );
double uxxyy_exact ( double x, double y );

int run_poisson (int num_thrs)
{
  bool converged;
  double diff;
  double dx;
  double dy;
  double error;
  double f[NX][NY];
  int i;
  int id;
  int itnew;
  int itold;
  int j;
  int jt;
  int jt_max = 20;
  int nx = NX;
  int ny = NY;
  double tolerance = 0.000001;
  double u[NX][NY];
  double u_norm;
  double udiff[NX][NY];
  double uexact[NX][NY];
  double unew[NX][NY];
  double unew_norm;
  double wtime;
  double x;
  double y;

  dx = 1.0 / ( double ) ( nx - 1 );
  dy = 1.0 / ( double ) ( ny - 1 );
//
//  Print a message.
//
  timestamp ( );
  
# pragma omp parallel
{
  id = omp_get_thread_num ( );
  
}
  
//
//  Set the right hand side array F.
//
  rhs ( nx, ny, f );
//
//  Set the initial solution estimate UNEW.
//  We are "allowed" to pick up the boundary conditions exactly.
//
  for ( j = 0; j < ny; j++ )
  {
    for ( i = 0; i < nx; i++ )
    {
      if ( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
      {
        unew[i][j] = f[i][j];
      }
      else
      {
        unew[i][j] = 0.0;
      }
    }
  }
  unew_norm = r8mat_rms ( nx, ny, unew );
//
//  Set up the exact solution UEXACT.
//
  for ( j = 0; j < ny; j++ )
  {
    y = ( double ) ( j ) / ( double ) ( ny - 1 );
    for ( i = 0; i < nx; i++ )
    {
      x = ( double ) ( i ) / ( double ) ( nx - 1 );
      uexact[i][j] = u_exact ( x, y );
    }
  }
  u_norm = r8mat_rms ( nx, ny, uexact );
//
//  Do the iteration.
//
  converged = false;

  for ( j = 0; j < ny; j++ )
  {
    for ( i = 0; i < nx; i++ )
    {
      udiff[i][j] = unew[i][j] - uexact[i][j];
    }
  }
  error = r8mat_rms ( nx, ny, udiff );
  wtime = omp_get_wtime ( );

  itnew = 0;

  for ( ; ; )
  {
    itold = itnew;
    itnew = itold + 500;
//
//  SWEEP carries out 500 Jacobi steps in parallel before we come
//  back to check for convergence.
//
    sweep ( nx, ny, dx, dy, f, itold, itnew, u, unew , num_thrs);
//
//  Check for convergence.
//
    u_norm = unew_norm;
    unew_norm = r8mat_rms ( nx, ny, unew );

    for ( j = 0; j < ny; j++ )
    {
      for ( i = 0; i < nx; i++ )
      {
        udiff[i][j] = unew[i][j] - u[i][j];
      }
    }
    diff = r8mat_rms ( nx, ny, udiff );

    for ( j = 0; j < ny; j++ )
    {
      for ( i = 0; i < nx; i++ )
      {
        udiff[i][j] = unew[i][j] - uexact[i][j];
      }
    }
    error = r8mat_rms ( nx, ny, udiff );

    if ( diff <= tolerance )
    {
      converged = true;
      break;
    }

  }
  wtime = omp_get_wtime ( ) - wtime;
  
//
//  Terminate.
//
  //timestamp ( );

  return 0;
}

double r8mat_rms ( int nx, int ny, double a[NX][NY] )
{
  int i;
  int j;
  double v;

  v = 0.0;

  for ( j = 0; j < ny; j++ )
  {
    for ( i = 0; i < nx; i++ )
    {
      v = v + a[i][j] * a[i][j];
    }
  }
  v = sqrt ( v / ( double ) ( nx * ny )  );

  return v;
}

void rhs ( int nx, int ny, double f[NX][NY] )
{
  double fnorm;
  int i;
  int j;
  double x;
  double y;
//
//  The "boundary" entries of F store the boundary values of the solution.
//  The "interior" entries of F store the right hand sides of the Poisson equation.
//
  for ( j = 0; j < ny; j++ )
  {
    y = ( double ) ( j ) / ( double ) ( ny - 1 );
    for ( i = 0; i < nx; i++ )
    {
      x = ( double ) ( i ) / ( double ) ( nx - 1 );
      if ( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
      {
        f[i][j] = u_exact ( x, y );
      }
      else
      {
        f[i][j] = - uxxyy_exact ( x, y );
      }
    }
  }

  fnorm = r8mat_rms ( nx, ny, f );

  return;
}

void sweep ( int nx, int ny, double dx, double dy, double f[NX][NY], 
  int itold, int itnew, double u[NX][NY], double unew[NX][NY] , int num_thrs)
{
  int i;
  int it;
  int j;

# pragma omp parallel num_threads(num_thrs) shared ( dx, dy, f, itnew, itold, nx, ny, u, unew ) private ( i, it, j )

  for ( it = itold + 1; it <= itnew; it++ )
  {
//
//  Save the current estimate.
//
# pragma omp for 
    for ( j = 0; j < ny; j++ )
    {
      for ( i = 0; i < nx; i++ )
      {
        u[i][j] = unew[i][j];
      }
    }
//
//  Compute a new estimate.
//
# pragma omp for
    for ( j = 0; j < ny; j++ )
    {
      for ( i = 0; i < nx; i++ )
      {
        if ( i == 0 || j == 0 || i == nx - 1 || j == ny - 1 )
        {
          unew[i][j] = f[i][j];
        }
        else
        { 
          unew[i][j] = 0.25 * ( 
            u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy );
        }
      }
    }

  }
  return;
}

double u_exact ( double x, double y )
{
  double pi = 3.141592653589793;
  double value;

  value = sin ( pi * x * y );

  return value;
}

double uxxyy_exact ( double x, double y )
{
  double pi = 3.141592653589793;
  double value;

  value = - pi * pi * ( x * x + y * y ) * sin ( pi * x * y );

  return value;
}

int run_circuit ( int num_thrs);
int circuit_value ( int n, int bvec[] );
void i4_to_bvec ( int i4, int n, int bvec[] );

//****************************************************************************80

int run_circuit (int num_thrs)

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for SATISFY_OPENMP.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    24 March 2009
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Michael Quinn,
//    Parallel Programming in C with MPI and OpenMP,
//    McGraw-Hill, 2004,
//    ISBN13: 978-0071232654,
//    LC: QA76.73.C15.Q55.
//
{
# define N 25

  int bvec[N];
  int i;
  int id;
  int ihi;
  int ihi2;
  int ilo;
  int ilo2;
  int j;
  int n = N;
  int solution_num;
  int solution_num_local;
  int thread_num;
  int value;
  double wtime;

  
  //timestamp ( );
 
//
//  Compute the number of binary vectors to check.
//
  ilo = 0;
  ihi = 1;
  for ( i = 1; i <= n; i++ )
  {
    ihi = ihi * 2;
  }
  
//
//  Processor ID takes the interval ILO2 <= I < IHI2.
//  Using the formulas below yields a set of nonintersecting intervals
//  which cover the original interval [ILO,IHI).
//
  thread_num=num_thrs;
  solution_num = 0;

  wtime = omp_get_wtime ( );

# pragma omp parallel \
  num_threads(num_thrs) \
  shared ( ihi, ilo, n, thread_num ) \
  private ( bvec, i, id, ihi2, ilo2, j, solution_num_local, value ) \
  reduction ( + : solution_num )
  {
    id = omp_get_thread_num ( );

    ilo2 = ( ( thread_num - id     ) * ilo   
           + (              id     ) * ihi ) 
           / ( thread_num          );

    ihi2 = ( ( thread_num - id - 1 ) * ilo   
           + (              id + 1 ) * ihi ) 
           / ( thread_num          );

    
//
//  Check every possible input vector.
//
    solution_num_local = 0;

    for ( i = ilo2; i < ihi2; i++ )
    {
      i4_to_bvec ( i, n, bvec );

      value = circuit_value ( n, bvec );

      if ( value == 1 )
      {
        solution_num_local = solution_num_local + 1;
 
      }
    }
    solution_num = solution_num + solution_num_local;
  }
  wtime = omp_get_wtime ( ) - wtime;
  
//
//  Terminate.
//
 
  //timestamp ( );

  return 0;
# undef N
}
//****************************************************************************80

int circuit_value ( int n, int bvec[] )

//****************************************************************************80
//
//  Purpose:
//
//    CIRCUIT_VALUE returns the value of a circuit for a given input set.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    20 March 2009
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Michael Quinn,
//    Parallel Programming in C with MPI and OpenMP,
//    McGraw-Hill, 2004,
//    ISBN13: 978-0071232654,
//    LC: QA76.73.C15.Q55.
//
//  Parameters:
//
//    Input, int N, the length of the input vector.
//
//    Input, int BVEC[N], the binary inputs.
//
//    Output, int CIRCUIT_VALUE, the output of the circuit.
//
{
  int value;

  value = 
       (  bvec[0]  ||  bvec[1]  )
    && ( !bvec[1]  || !bvec[3]  )
    && (  bvec[2]  ||  bvec[3]  )
    && ( !bvec[3]  || !bvec[4]  )
    && (  bvec[4]  || !bvec[5]  )
    && (  bvec[5]  || !bvec[6]  )
    && (  bvec[5]  ||  bvec[6]  )
    && (  bvec[6]  || !bvec[15] )
    && (  bvec[7]  || !bvec[8]  )
    && ( !bvec[7]  || !bvec[13] )
    && (  bvec[8]  ||  bvec[9]  )
    && (  bvec[8]  || !bvec[9]  )
    && ( !bvec[9]  || !bvec[10] )
    && (  bvec[9]  ||  bvec[11] )
    && (  bvec[10] ||  bvec[11] )
    && (  bvec[12] ||  bvec[13] )
    && (  bvec[13] || !bvec[14] )
    && (  bvec[14] ||  bvec[15] )
    && (  bvec[14] ||  bvec[16] )
    && (  bvec[17] ||  bvec[1]  )
    && (  bvec[18] || !bvec[0]  )
    && (  bvec[19] ||  bvec[1]  )
    && (  bvec[19] || !bvec[18] )
    && ( !bvec[19] || !bvec[9]  )
    && (  bvec[0]  ||  bvec[17] )
    && ( !bvec[1]  ||  bvec[20] )
    && ( !bvec[21] ||  bvec[20] )
    && ( !bvec[22] ||  bvec[20] )
    && ( !bvec[21] || !bvec[20] )
    && (  bvec[22] || !bvec[20] );

  return value;
}
//****************************************************************************80

void i4_to_bvec ( int i4, int n, int bvec[] )

//****************************************************************************80
//
//  Purpose:
//
//    I4_TO_BVEC converts an integer into a binary vector.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    20 March 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I4, the integer.
//
//    Input, int N, the dimension of the vector.
//
//    Output, int BVEC[N], the vector of binary remainders.
//
{
  int i;

  for ( i = n - 1; 0 <= i; i-- )
  {
    bvec[i] = i4 % 2;
    i4 = i4 / 2;
  }

  return;
}

/*  Helper function for retrieving runtime information using SIGAR
 *  General CPU, MEM , VM info and process specific CPU, MEM , VM info
 */
 
 /* Get memory usage info for a process and fill the appropriate record attributes
  * Parametars:
  *  - sigar_pid_t pid : the process in the SIGAR wrapper for pid_t
  *  - dataset_record * record : a pointer to the record to be modified
  */ 
void pid_mem_info (sigar_pid_t pid, dataset_record * record){
	
	sigar_t *sigarproclist;
    sigar_open(&sigarproclist);
    sigar_proc_mem_t mem_pid_info;
	int res_status = sigar_proc_mem_get(sigarproclist, pid, &mem_pid_info);
    if (res_status == SIGAR_OK){
		// divide with total after returning 
		record->pid_page_faults = mem_pid_info.page_faults;
		record->pid_ram_usage = mem_pid_info.size;
		/*
		cout<<"RECORD - pid_page_faults:"<<mem_pid_info.page_faults<<endl;
		cout<<"RECORD - pid_ram_usage:"<<record->pid_ram_usage<<endl;
		
		cout <<"minor_faults:"<<mem_pid_info.minor_faults<< endl;
		cout <<"major_faults:"<<mem_pid_info.major_faults<< endl;
		cout <<"page_faults:"<<mem_pid_info.page_faults<< endl;
		cout <<"Size:"<<mem_pid_info.size<< endl;
		cout <<"Share:"<<mem_pid_info.share<< endl;
		*/
	}
	
	sigar_close(sigarproclist);
}

/* Get vm usage info for a process and fill the appropriate record attributes
  * Parametars:
  *  - sigar_pid_t pid : the process in the SIGAR wrapper for pid_t
  *  - dataset_record * record : a pointer to the record to be 
  */ 
void pid_vm_info(sigar_pid_t pid, dataset_record * record){
	sigar_t *sigarproclist;
    sigar_open(&sigarproclist);
    sigar_proc_cumulative_disk_io_t vm_pid_info;
	int res_status = sigar_proc_cumulative_disk_io_get(sigarproclist, pid, &vm_pid_info);
	if (res_status == SIGAR_OK){
		if (vm_pid_info.bytes_total == 0){
		record-> pid_vm_read_usage = (double) vm_pid_info.bytes_read;
		record-> pid_vm_write_usage = (double) vm_pid_info.bytes_written;
		}
		else {
		record-> pid_vm_read_usage = (double) vm_pid_info.bytes_read/vm_pid_info.bytes_total;
		record-> pid_vm_write_usage = (double) vm_pid_info.bytes_written/vm_pid_info.bytes_total;
		}
		/*
		cout<<"RECORD - pid_vm_read_usage: "<<record-> pid_vm_read_usage<<endl;
		cout<<"RECORD - pid_vm_write_usage: "<<record-> pid_vm_write_usage<<endl;
		
		cout<<"bytes_read"<<vm_pid_info.bytes_read<<endl;
		cout<<"bytes_written"<<vm_pid_info.bytes_written<<endl;
	    cout<<"bytes_total"<<vm_pid_info.bytes_total<<endl;
	    */
	}
	
	sigar_close(sigarproclist);

}

/* Get cpu usage info for the current process and fill the appropriate record attributes
  * It takes two snapshots with a time interval of 10*5 mictroseconds and retrives the state
  * Calls the pid_mem_info function and the pid_vm_info function for the current process
  * Parametars:
  *  - dataset_record * record : a pointer to the record to be modified
  */ 
void get_pid_cpu(dataset_record * record){
	// Init CIGAR 
	sigar_t *sigarcpulist;
	sigar_cpu_info_list_t cpulist;
	sigar_open(&sigarcpulist);
	sigar_cpu_info_list_get(sigarcpulist, &cpulist);
	sigar_close(sigarcpulist);
	//(void)sigar_cpu_total_count(sigarcpulist);
	//cout<<"Total num processors:"<<sigarcpulist->ncpu<<endl;

	float percent;
    
   // while(1){
		{
		float percent;
		sigar_t *sigarproclist;
		sigar_proc_list_t proclist;
		sigar_open(&sigarproclist);
		sigar_proc_list_get(sigarproclist, &proclist);
		
		for (size_t i = 0; i < proclist.number; i++){
			sigar_proc_cpu_t cpu;
			sigar_proc_cpu_get(sigarproclist, proclist.data[i], &cpu);
		}
		
		unsigned int microseconds = 1000000;
		usleep(microseconds);
		
		double curr_pid_cpu = 0;
		// count of processes in running and blocking state
		int per_running = 0; 
		int per_blocking = 0;
		int total = 0;
		// number of page faults , later to be used as a denominator for the page faults ration of the current process
		int page_faults_total=0;
		
		// Iteration of each running process
		for(size_t i = 0; i<proclist.number;i++){
			total++;
			sigar_proc_cpu_t cpu;
			int status = sigar_proc_cpu_get(sigarproclist, proclist.data[i], &cpu);
			
			if (status == SIGAR_OK){
				
				sigar_proc_state_t procstate;
				sigar_proc_state_get(sigarproclist, proclist.data[i], &procstate);
				percent = cpu.percent * 100 / cpulist.size;
				
				// check if the process is the current process, if yes fetch mem and vm info
				if((int)procstate.ppid == (int)getppid()){
					//cout<<"Current process cpu usage info:"<<endl;
					//cout<<procstate.name << percent<<"% "<<" priority:"<<procstate.priority<<" ppid:"<<procstate.ppid<<" state:"<<procstate.state<<endl;
					curr_pid_cpu = percent;
					record -> pid_cpu_usage = percent;
					
					// calcuation of cumulate page_faults
					sigar_proc_mem_t procmem_curr;
					int status = sigar_proc_mem_get(sigarproclist, procstate.ppid, &procmem_curr);
                    if(status == SIGAR_OK){
						page_faults_total += procmem_curr.page_faults;
					}
					
					// retieve mem info for this process
					//cout<<"about to retrive mem info for this pid"<<endl;
					pid_mem_info(procstate.ppid, record);
					
					// retieve vm/swap info for this process
					//cout<<"about to retrieve vm info fot this pid"<<endl;
					pid_vm_info(procstate.ppid, record);
					
				}
					
				if((char)procstate.state == 'R'){
					per_running ++;
				}
				else if((char)procstate.state == 'S'){
					per_blocking ++;
				}
			}
		}
		
		//cout<<"Percent running proccesses:"<<per_running<<endl;
		//cout<<"Percent blocking proccesses:"<<per_blocking<<endl;
		//cout<<"Total proccesses:"<<total<<endl;
		
		if (total == 0){
		   total = 1;
		}
		double percent_running = (double)per_running/ total;
		record->procs_running = percent_running;
		//cout<<"RECORD - procs_running"<<record->procs_running<<endl;
		//cout<<"Percent running proccesses:"<<percent_running<<endl;
		double percent_blocking = (double)per_blocking /total;
		record->procs_blocked = percent_blocking;
		//cout<<"RECORD - procs_blocking"<<record->procs_blocked<<endl;

		//cout<<"Percent blocking proccesses:"<<percent_blocking<<endl;
		if(page_faults_total==0){
		page_faults_total=1;
		}
		record->pid_page_faults = (double)record->pid_page_faults/page_faults_total;
		//cout<<"RECORD - pid_page_faults, init: "<<record->pid_page_faults<<endl;
		// divide just obtained pid_ram_usage, ram_usage already fetched!
		if (record->ram_usage == 0){
				record->pid_ram_usage = (double) record->pid_ram_usage;
		}
		else {
		record->pid_ram_usage = (double) record->pid_ram_usage/ record->ram_usage;
		}
		//cout<<"RECORD - pid_ram_usage, init:"<< record->pid_ram_usage<<endl;
		sigar_close(sigarproclist);
	}
}

/* Get total memory usage info and modify the appropriate record attributes
  * Parametars:
  *  - dataset_record * record : a pointer to the record to be modified
  */ 
void get_mem_info(dataset_record * record){
	sigar_t *sigarproclist;
    sigar_open(&sigarproclist);
    sigar_mem_t mem_info;
    int res_status = sigar_mem_get(sigarproclist,&mem_info);
    if (res_status == SIGAR_OK){
		if (mem_info.total==0){
			record->ram_usage = (double)mem_info.actual_used;
		}
		else {
		record->ram_usage = (double)mem_info.actual_used/mem_info.total;
		}
		/*
		cout<<"RECORD - ram_usage:"<<record->ram_usage<<endl;
		cout <<"Total:"<<mem_info.total<< endl;
		cout <<"Free:"<<mem_info.free<< endl;
		cout <<"Used:"<<mem_info.used<< endl;
		cout <<"Actual used:"<<mem_info.actual_used<< endl;
		cout <<"Actual free:"<<mem_info.actual_free<< endl;
		cout <<"Actual total:"<<mem_info.total<< endl;
		*/
	}
	sigar_close(sigarproclist);
}

/* Get total vm usage info and modify the appropriate record attributes
  * Parametars:
  *  - dataset_record * record : a pointer to the record to be modified
  */ 
void get_vm_info(dataset_record * record){
	sigar_t *sigarproclist;
    sigar_open(&sigarproclist);
	sigar_swap_t vm_info;
	int res_status = sigar_swap_get(sigarproclist, &vm_info);
	if (res_status == SIGAR_OK){
		if (vm_info.total == 0){
		 record->vm_usage = (double)vm_info.used;
		}
		else {
		record->vm_usage = (double)vm_info.used/vm_info.total;
		}
		/*
		cout<<"RECORD - vm_usage: "<<record->ram_usage<<endl;
		cout<<"Total vm: "<<vm_info.total<<endl;
		cout<<"Free vm: "<<vm_info.free<<endl;
		cout<<"Used vm: "<<vm_info.used<<endl;
		*/
	}
	sigar_close(sigarproclist);
}

/* Get total cpu info on all cpus and modify the appropriate record attributes
  * Parametars:
  *  - dataset_record * record : a pointer to the record to be modified
  */ 
static double cpu_used = 0;

static unsigned long long cpu_last_used = 0;
static unsigned long long cpu_last_total = 0;
static unsigned long long cpu_last_idle = 0;
static unsigned long long cpu_new_used = 0;
static unsigned long long cpu_new_total = 0;
static unsigned long long cpu_new_idle = 0;
static void* timerid =NULL;
static int cpu_sample_interval = 10000;//1s

int get_cpu_list_info(dataset_record * record)
{
    int status, i;
    sigar_cpu_list_t cpulist;

    sigar_t *sigar;
    sigar_open(&sigar);

    status = sigar_cpu_list_get(sigar, &cpulist);

    if (status != SIGAR_OK)
    {
        printf("cpu_list error: %d (%s)\n",
               status, sigar_strerror(sigar, status));
        return -1;
    }
    //printf("sigar ok\n");
    //printf("sigar ok\n");
    //if (cpu_last_used == 0)
    {
        cpu_last_used = cpu_new_used;
        cpu_last_total = cpu_new_total;
        cpu_last_idle = cpu_new_idle;
        //cpu_sample_time = getustime();
    }
    cpu_new_used = 0;
    cpu_new_total = 0;
    cpu_new_idle = 0;
    
    for (i=0; i<cpulist.number; i++)
    {
		
        sigar_cpu_t cpu = cpulist.data[i];   
        //printf("%d:user = %llu,sys= %llu,nice= %llu,idle= %llu, wait= %llu,irq= %llu,soft_irq= %llu,stolen= %llu,total= %llu END\n",
                  //i, cpu.user, cpu.sys, cpu.nice,cpu.idle,cpu.wait, cpu.irq, cpu.soft_irq,cpu.stolen,cpu.total);
        
        cpu_new_used += cpu.user + cpu.sys;
        cpu_new_total += cpu.total;
        cpu_new_idle += cpu.idle;
    }

    double used_diff = cpu_new_used - cpu_last_used;
    double total_diff = cpu_new_total- cpu_last_total;
    double idle_diff = cpu_new_idle - cpu_last_idle;
    double used = 0;
    double idle = 0;
    if (total_diff != 0)
    {
        used = 100.0 * used_diff/total_diff;
        idle = 100.0 * idle_diff/total_diff;
        if (used < 0)
        {
            used = 0.0 - used;
        }
        if (used >= 100.0)
        {
            used = 99;
        }

        cpu_used = used;
        record->lst_cpu_usage = cpu_used;
    }

    //double usage = toMillis(total_diff)/((getustime()-cpu_sample_time)*0.001);
   // printf("used_diff=%f,total_diff=%f, cpu_used=%f, idle_percent= %f",used_diff, total_diff, cpu_used, idle);
    sigar_cpu_list_destroy(sigar, &cpulist);

    return 0;
}

void get_cpu_list_info1(dataset_record * record){
	sigar_t *sigar;
	sigar_open(&sigar);
	sigar_cpu_list_t sigar_cpu_list;
	int status = sigar_cpu_list_get(sigar, & sigar_cpu_list);
	
	if (status != SIGAR_OK) return;
	
	unsigned long long cpu_list_prev_total=0;
	unsigned long long cpu_list_prev_user=0;
	unsigned long long cpu_list_prev_sys=0;
	unsigned long long cpu_list_prev_idle=0;
	unsigned long long cpu_list_prev_nice=0;
	unsigned long long cpu_list_prev_wait=0;
	unsigned long long cpu_list_prev_irq=0;
	unsigned long long cpu_list_prev_soft_irq=0;
	unsigned long long cpu_list_prev_stolen=0;
	unsigned long long cpu_list_next_total=0;
	unsigned long long cpu_list_next_user=0;
	unsigned long long cpu_list_next_sys=0;
	unsigned long long cpu_list_next_idle=0;
	unsigned long long cpu_list_next_nice=0;
	unsigned long long cpu_list_next_wait=0;
	unsigned long long cpu_list_next_irq=0;
	unsigned long long cpu_list_next_soft_irq=0;
	unsigned long long cpu_list_next_stolen=0;

	sigar_cpu_t tmp_cpu; 
	for(int i=0;i<sigar_cpu_list.number;i++){
		tmp_cpu = sigar_cpu_list.data[i];
		//cout<<"CPU "<<i<<" :::::"<<endl;
		cpu_list_prev_total += tmp_cpu.total;
		//cout<<"prev cpu total"<<tmp_cpu.total<<endl;
        cpu_list_prev_user += tmp_cpu.user;
        //cout<<"prev cpu user"<<tmp_cpu.user<<endl;
        cpu_list_prev_sys += tmp_cpu.sys;
        //cout<<"prev cpu sys"<<tmp_cpu.sys<<endl;
        cpu_list_prev_idle += tmp_cpu.idle;
        //cout<<"prev cpu idle"<<tmp_cpu.idle<<endl;
        cpu_list_prev_nice += tmp_cpu.nice;
        //cout<<"prev cpu nice"<<tmp_cpu.nice<<endl;
        cpu_list_prev_wait += tmp_cpu.wait;
         //cout<<"prev cpu wait"<<tmp_cpu.wait<<endl;
        cpu_list_prev_irq += tmp_cpu.irq;
		cpu_list_prev_soft_irq += tmp_cpu.soft_irq;
        cpu_list_prev_stolen += tmp_cpu.stolen;
	}
	
	unsigned int microsec = 1000000;
	usleep(microsec);
	
	sigar_close(sigar);
	// second snapshot
	//sigar_t *sigar;
	sigar_t *sigar1;
	sigar_open(&sigar1);
	sigar_cpu_list_t  sigar_cpu_list1;
	int status1 = sigar_cpu_list_get(sigar, & sigar_cpu_list1);
	
	if(status1!= SIGAR_OK) return;
	
	for(int i=0;i<sigar_cpu_list.number;i++){
		tmp_cpu = sigar_cpu_list.data[i];
		//cout<<"CPU "<<i<<endl;
		cpu_list_next_total += tmp_cpu.total;
		//cout<<"next cpu total"<<tmp_cpu.total<<endl;
        cpu_list_next_user += tmp_cpu.user;
        //cout<<"next cpu user"<<tmp_cpu.user<<endl;
        cpu_list_next_sys += tmp_cpu.sys;
        //cout<<"next cpu sys"<<tmp_cpu.sys<<endl;
        cpu_list_next_idle += tmp_cpu.idle;
        //cout<<"next cpu idle"<<tmp_cpu.idle<<endl;
        cpu_list_next_nice += tmp_cpu.nice;
        //cout<<"next cpu nice"<<tmp_cpu.nice<<endl;
        cpu_list_next_wait += tmp_cpu.wait;
        //cout<<"next cpu wait"<<tmp_cpu.wait<<endl;
        cpu_list_next_irq += tmp_cpu.irq;
        //cout<<"next cpu irq"<<tmp_cpu.irq<<endl;
		cpu_list_next_soft_irq += tmp_cpu.soft_irq;
        cpu_list_next_stolen += tmp_cpu.stolen;
	}
	
	record->lst_total = (cpu_list_next_total-cpu_list_prev_total);
	printf("cpu_list_next_total = %llu",cpu_list_next_total);
	printf("cpu_list_prev_total = %llu",cpu_list_prev_total);
	unsigned long long tmp_diff= cpu_list_next_total-cpu_list_prev_total;
	printf("diff = %llu",tmp_diff);
	printf("cpu_list_next_idle = %llu",cpu_list_next_idle);
	printf("cpu_list_prev_idle = %llu",cpu_list_prev_idle);
	tmp_diff= (cpu_list_next_idle-cpu_list_prev_idle);
	printf("diff = %llu",tmp_diff);
	//cout<<"record->lst_total:"<<record->lst_total<<", diff:"<<(cpu_list_next_total-cpu_list_prev_total)<<endl;
	
	record->lst_idle = (cpu_list_next_idle-cpu_list_prev_idle);
	//cout<<"record->idle:"<<record->lst_idle<<endl;
	printf("cpu_list_next_idle = %llu",cpu_list_next_idle);
	printf("cpu_list_prev_idle = %llu",cpu_list_prev_idle);
	tmp_diff= (cpu_list_next_idle-cpu_list_prev_idle);
	printf("diff = %llu",tmp_diff);
	
	record->lst_sys = cpu_list_next_sys-cpu_list_prev_sys;
	record->lst_user = cpu_list_next_user-cpu_list_prev_user;
	record->lst_irq = cpu_list_next_irq-cpu_list_prev_irq;
	record->lst_soft_irq = cpu_list_next_soft_irq-cpu_list_prev_soft_irq;
	record->lst_nice = cpu_list_next_nice-cpu_list_prev_nice;
	record->lst_wait = cpu_list_next_wait-cpu_list_prev_wait;
	record->lst_stolen = cpu_list_next_stolen-cpu_list_prev_stolen;
	record->lst_cpu_usage = ((record->lst_user+record->lst_sys)/record->lst_total)*100;
	sigar_close(sigar1);
}

/* Get total current cpu usage info and modify the appropriate record attributes
  * Parametars:
  *  - dataset_record * record : a pointer to the record to be modified
  */ 
void get_cpu_info(dataset_record * record){
	
	sigar_t *sigar_cpu;
	sigar_cpu_t old;
	sigar_cpu_t current;

	sigar_open(&sigar_cpu);
	sigar_cpu_get(sigar_cpu, &old);

	sigar_cpu_perc_t perc;
	unsigned int microseconds=100000;
	sigar_cpu_get(sigar_cpu, &current);
    sigar_cpu_perc_calculate(&old, &current, &perc);
	old = current;
	usleep(microseconds);
	sigar_cpu_get(sigar_cpu, &current);
	sigar_cpu_perc_calculate(&old, &current, &perc);
	
	record->wait= perc.wait;
	//cout<<"RECORD - wait: "<<record->wait<<endl;
	record->steal_time = perc.stolen;
	//cout<<"RECORD - steal_time: "<<record->steal_time<<endl;
	record->user = perc.user;
	//cout<<"RECORD - user: "<<record->user<<endl;
	record->sys = perc.sys;
	//cout<<"RECORD - sys: "<<record->sys<<endl;
	record->nice = perc.nice;
	//cout<<"RECORD - nice: "<<record->nice<<endl;
	record->idle = perc.idle;
	//cout<<"RECORD - idle: "<<record->idle<<endl;
	record->irq = perc.irq;
	//cout<<"RECORD - irq: "<<record->irq<<endl;
	record->cpu_usage = perc.combined;
	/*
	cout<<"RECORD - cpu_usage: "<<record->cpu_usage<<endl;
	std::cout << "CPU soft irq" << perc.soft_irq << "%\n";
	std::cout << "CPU wait" << perc.wait << "%\n";
	std::cout << "CPU stolen" << perc.stolen << "%\n";
	std::cout << "CPU irq" << perc.irq << "%\n";
	std::cout << "CPU idle" << perc.idle << "%\n";
	std::cout << "CPU nice" << perc.nice << "%\n";
	std::cout << "CPU user" << perc.user << "%\n";
	std::cout << "CPU sys " << perc.sys << "%\n";
    std::cout << "CPU combined " << perc.combined * 100 << "%\n";*/
}

/* Function that retrieves data for a record
 * Parametars:
 *  - dataset_record * record : a pointer to the record to be modified 
 */
 void get_runtime_stats(dataset_record * record){
  get_mem_info(record);
  get_vm_info(record);
  get_cpu_info(record);
  get_cpu_list_info(record);
  get_pid_cpu(record);
}
// Functions related to the random forest 

// This function reads data and responses from the file <filename>
static int
read_num_class_data( const char* filename, int var_count,
                     CvMat** data, CvMat** responses )
{
	const int M = 1024;
    FILE* f = fopen( filename, "rt" );
    CvMemStorage* storage;
    CvSeq* seq;
    char buf[M+2];
    float* el_ptr;
    CvSeqReader reader;
    int i, j;

    if( !f )
        return 0;

    el_ptr = new float[var_count+1];
    storage = cvCreateMemStorage();
    seq = cvCreateSeq( 0, sizeof(*seq), (var_count+1)*sizeof(float), storage );

    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        el_ptr[0] = buf[0];
        ptr = buf+2;
        for( i = 1; i <= var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", el_ptr + i, &n );
            ptr += n + 1;
        }
        if( i <= var_count )
            break;
        cvSeqPush( seq, el_ptr );
    }
    fclose(f);

    *data = cvCreateMat( seq->total, var_count, CV_32F );
    *responses = cvCreateMat( seq->total, 1, CV_32F );

    cvStartReadSeq( seq, &reader );

    for( i = 0; i < seq->total; i++ )
    {
        const float* sdata = (float*)reader.ptr + 1;
        float* ddata = data[0]->data.fl + var_count*i;
        float* dr = responses[0]->data.fl + i;

        for( j = 0; j < var_count; j++ )
            ddata[j] = sdata[j];
        *dr = sdata[-1];
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    cvReleaseMemStorage( &storage );
    delete[] el_ptr;
    return 1;
}

 static CvRTrees forest;
 
 /*
  * Function that returns the predicted optimal number of threads for a current runtime state
  * */
 double predict_for_sample(dataset_record * record){
	// predict for dataset 1, 11 vars
	/*
	double array [11];
	for (int i=0;i<11;i++){
		array[i]= 0.1f;
	}
	
	array[0]=0.0f;
	array[1]=(double)record->cpu_usage;
	array[2]=(double)record->pid_cpu_usage;
	array[3]=(double)record->lst_cpu_usage;
	array[4]=(double)record->ram_usage;
	array[5]=(double)record->pid_page_faults;
	array[6]=(double)record->procs_blocked;
	array[7]=(double)record->user;
	array[8]=(double)record->sys;
	array[9]=(double)record->idle;
	array[10]=(double)record->irq;
	*/
	
	double array [28];
	for (int i=0;i<28;i++){
		array[i]= 0.1f;
	}
	
	array[0]=4.0f;
	array[1]=(double)record->cpu_usage;
	array[2]=(double)record->pid_cpu_usage;
	array[3]=(double)record->lst_cpu_usage;
	array[4]=(double)record->ram_usage;
	array[5]=(double)record->pid_page_faults;
	array[6]=(double)record->pid_ram_usage;
	array[7]=(double)record->procs_blocked;
	array[8]=(double)record->processes;
	array[9]=(double)record->procs_running;
	array[10]=(double)record->procs_blocked;
	array[11]=(double)record->user;
	array[12]=(double)record->sys;
	array[13]=(double)record->nice;
	array[14]=(double)record->idle;
	array[15]=(double)record->wait;
	array[16]=(double)record->irq;
	array[17]=(double)record->soft_irq;  
	array[18]=(double)record->lst_stolen; 
	array[19]=(double)record->lst_user;
	array[20]=(double)record->lst_sys; 
	array[21]=(double)record->lst_nice;
	array[22]=(double)record->lst_idle; 
	array[23]=(double)record->lst_wait;
	array[24]=(double)record->lst_irq; 
	array[25]=(double)record->lst_soft_irq; 
	array[26]=(double)record->lst_total; 
	array[27]=(double)record->lst_cpu_usage;
     
   //  array[0]=4.0f;
   
   /*
   double array [8];
	for (int i=0;i<8;i++){
		array[i]= 0.1f;
	}
	array[0]=(double)record->pid_cpu_usage;
	array[1]=(double)record->lst_cpu_usage;
	array[2]=(double)record->lst_nice;
	array[3]=(double)record->lst_idle; 
	array[4]=(double)record->lst_wait;
	array[5]=(double)record->lst_irq; 
	array[6]=(double)record->lst_soft_irq;
	array[7]=(double)record->lst_cpu_usage;
	*/
	/*
	double array [14];
	for (int i=0;i<14;i++){
		array[i]= 0.1f;
	}
	
	array[0]=(double)record->num_threads;
	array[1]=(double)record->pid_ram_usage;
	array[2]=(double)record->ram_usage;
	array[3]=(double)record->pid_cpu_usage; 
	array[4]=(double)record->cpu_usage;
	array[5]=(double)record->procs_blocked;
	array[6]=(double)record->nice;
	array[7]=(double)record->idle;
	array[8]=(double)record->user;
	array[9]=(double)record->sys;
	array[10]=(double)record->wait;
	array[11]=(double)record->irq;
	array[12]=(double)record->soft_irq;
	array[13]=(double)record->lst_cpu_usage;
  */
	cv::Mat mat(1, 28, CV_32FC1, &array);

	double r = (double)forest.predict(mat);
    return fabs(r);

}
 
// Build the rtreess , random forest classifier 
int build_rtrees_classifier( char* data_filename,
    char* filename_to_save, char* filename_to_load )
{
    CvMat* data = 0;
    CvMat* responses = 0;
    CvMat* var_type = 0;
    CvMat* sample_idx = 0;

    int ok = read_num_class_data( data_filename, 28 , &data, &responses );
    int nsamples_all = 0, ntrain_samples = 0;
    int i = 0;
    double train_hr = 0, test_hr = 0;
   
    CvMat* var_importance = 0;

    if( !ok )
    {
        printf( "Could not read the database %s\n", data_filename );
        return -1;
    }

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load Random Trees classifier
    if( filename_to_load && forest_created)
    {
        // load classifier from the specified file
        forest.load( filename_to_load );
        
        ntrain_samples = 0;
        if( forest.get_tree_count() == 0 )
        {
            printf( "Could not read the classifier %s\n", filename_to_load );
            return -1;
        }
        printf( "The classifier %s is loaded.\n", filename_to_load );
    }
    else
    {
        // create classifier by using <data> and <responses>
        printf( "Training the classifier ...\n");
		printf("Num attributes per sample: %d",data->cols + 1);
        // 1. create type mask
        var_type = cvCreateMat( data->cols + 1, 1, CV_8U );
        cvSet( var_type, cvScalarAll(CV_VAR_ORDERED) );
        cvSetReal1D( var_type, data->cols, CV_VAR_CATEGORICAL );

        // 2. create sample_idx
        sample_idx = cvCreateMat( 1, nsamples_all, CV_8UC1 );
        {
            CvMat mat;
            cvGetCols( sample_idx, &mat, 0, ntrain_samples );
            cvSet( &mat, cvRealScalar(1) );

            cvGetCols( sample_idx, &mat, ntrain_samples, nsamples_all );
            cvSetZero( &mat );
        }

        // 3. train classifier
        forest.train( data, CV_ROW_SAMPLE, responses, 0, sample_idx, var_type, 0,
            CvRTParams(25,10,0,false,25,0,true,4,50,0.01f,CV_TERMCRIT_ITER));
        printf( "\n");
    }

    // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        double r;
        CvMat sample;
        cvGetRow( data, &sample, i );

        r = forest.predict( &sample );
        
        r = fabs((double)r - responses->data.fl[i]) <= FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all-ntrain_samples);
    train_hr /= (double)ntrain_samples;
    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    printf( "Number of trees: %d\n", forest.get_tree_count() );

    // Print variable importance
    var_importance = (CvMat*)forest.get_var_importance();
    if( var_importance )
    {
        double rt_imp_sum = cvSum( var_importance ).val[0];
        printf("var#\timportance (in %%):\n");
        for( i = 0; i < var_importance->cols; i++ )
            printf( "%-2d\t%-4.1f\n", i,
            100.f*var_importance->data.fl[i]/rt_imp_sum);
    }

    //Print some proximitites
    /*
    printf( "Proximities between some samples corresponding to the letter 'T':\n" );
    {
        CvMat sample1, sample2;
        const int pairs[][2] = {{0,103}, {0,106}, {106,103}, {-1,-1}};

        for( i = 0; pairs[i][0] >= 0; i++ )
        {
            cvGetRow( data, &sample1, pairs[i][0] );
            cvGetRow( data, &sample2, pairs[i][1] );
            printf( "proximity(%d,%d) = %.1f%%\n", pairs[i][0], pairs[i][1],
                forest.get_proximity( &sample1, &sample2 )*100. );
        }
    }
    */

    // Save Random Trees classifier to file if needed
    if( filename_to_save )
        forest.save( filename_to_save );

    cvReleaseMat( &sample_idx );
    cvReleaseMat( &var_type );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}

// Build the boost classifier 
static
int build_boost_classifier( char* data_filename,
    char* filename_to_save, char* filename_to_load )
{
    const int class_count = 26;
    CvMat* data = 0;
    CvMat* responses = 0;
    CvMat* var_type = 0;
    CvMat* temp_sample = 0;
    CvMat* weak_responses = 0;

    int ok = read_num_class_data( data_filename, 11, &data, &responses );
    int nsamples_all = 0, ntrain_samples = 0;
    int var_count;
    int i, j, k;
    double train_hr = 0, test_hr = 0;
    CvBoost boost;

    if( !ok )
    {
        printf( "Could not read the database %s\n", data_filename );
        return -1;
    }

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.5);
    var_count = data->cols;

    // Create or load Boosted Tree classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        boost.load( filename_to_load );
        ntrain_samples = 0;
        if( !boost.get_weak_predictors() )
        {
            printf( "Could not read the classifier %s\n", filename_to_load );
            return -1;
        }
        printf( "The classifier %s is loaded.\n", filename_to_load );
    }
    else
    {
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //
        // As currently boosted tree classifier in MLL can only be trained
        // for 2-class problems, we transform the training database by
        // "unrolling" each training sample as many times as the number of
        // classes (26) that we have.
        //
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        CvMat* new_data = cvCreateMat( ntrain_samples*class_count, var_count + 1, CV_32F );
        CvMat* new_responses = cvCreateMat( ntrain_samples*class_count, 1, CV_32S );

        // 1. unroll the database type mask
        printf( "Unrolling the database...\n");
        for( i = 0; i < ntrain_samples; i++ )
        {
            float* data_row = (float*)(data->data.ptr + data->step*i);
            for( j = 0; j < class_count; j++ )
            {
                float* new_data_row = (float*)(new_data->data.ptr +
                                new_data->step*(i*class_count+j));
                for( k = 0; k < var_count; k++ )
                    new_data_row[k] = data_row[k];
                new_data_row[var_count] = (float)j;
                new_responses->data.i[i*class_count + j] = responses->data.fl[i] == j+'A';
            }
        }

        // 2. create type mask
        var_type = cvCreateMat( var_count + 2, 1, CV_8U );
        cvSet( var_type, cvScalarAll(CV_VAR_ORDERED) );
        // the last indicator variable, as well
        // as the new (binary) response are categorical
        cvSetReal1D( var_type, var_count, CV_VAR_CATEGORICAL );
        cvSetReal1D( var_type, var_count+1, CV_VAR_CATEGORICAL );

        // 3. train classifier
        printf( "Training the classifier (may take a few minutes)...\n");
        boost.train( new_data, CV_ROW_SAMPLE, new_responses, 0, 0, var_type, 0,
            CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 ));
        cvReleaseMat( &new_data );
        cvReleaseMat( &new_responses );
        printf("\n");
    }

    temp_sample = cvCreateMat( 1, var_count + 1, CV_32F );
    weak_responses = cvCreateMat( 1, boost.get_weak_predictors()->total, CV_32F );

    // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        int best_class = 0;
        double max_sum = -DBL_MAX;
        double r;
        CvMat sample;
        cvGetRow( data, &sample, i );
        for( k = 0; k < var_count; k++ )
            temp_sample->data.fl[k] = sample.data.fl[k];

        for( j = 0; j < class_count; j++ )
        {
            temp_sample->data.fl[var_count] = (float)j;
            boost.predict( temp_sample, 0, weak_responses );
            double sum = cvSum( weak_responses ).val[0];
            if( max_sum < sum )
            {
                max_sum = sum;
                best_class = j + 'A';
            }
        }

        r = fabs(best_class - responses->data.fl[i]) < FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all-ntrain_samples);
    train_hr /= (double)ntrain_samples;
    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    printf( "Number of trees: %d\n", boost.get_weak_predictors()->total );

    // Save classifier to file if needed
    if( filename_to_save )
        boost.save( filename_to_save );

    cvReleaseMat( &temp_sample );
    cvReleaseMat( &weak_responses );
    cvReleaseMat( &var_type );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}

// Build the mlp classifier 
static
int build_mlp_classifier( char* data_filename,
    char* filename_to_save, char* filename_to_load )
{
    const int class_count = 26;
    CvMat* data = 0;
    CvMat train_data;
    CvMat* responses = 0;
    CvMat* mlp_response = 0;

    int ok = read_num_class_data( data_filename, 12, &data, &responses );
    int nsamples_all = 0, ntrain_samples = 0;
    int i, j;
    double train_hr = 0, test_hr = 0;
    CvANN_MLP mlp;

    if( !ok )
    {
        printf( "Could not read the database %s\n", data_filename );
        return -1;
    }

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load MLP classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        mlp.load( filename_to_load );
        ntrain_samples = 0;
        if( !mlp.get_layer_count() )
        {
            printf( "Could not read the classifier %s\n", filename_to_load );
            return -1;
        }
        printf( "The classifier %s is loaded.\n", filename_to_load );
    }
    else
    {
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //
        // MLP does not support categorical variables by explicitly.
        // So, instead of the output class label, we will use
        // a binary vector of <class_count> components for training and,
        // therefore, MLP will give us a vector of "probabilities" at the
        // prediction stage
        //
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        CvMat* new_responses = cvCreateMat( ntrain_samples, class_count, CV_32F );

        // 1. unroll the responses
        printf( "Unrolling the responses...\n");
        for( i = 0; i < ntrain_samples; i++ )
        {
            int cls_label = cvRound(responses->data.fl[i]) - 'A';
            float* bit_vec = (float*)(new_responses->data.ptr + i*new_responses->step);
            for( j = 0; j < class_count; j++ )
                bit_vec[j] = 0.f;
            bit_vec[cls_label] = 1.f;
        }
        cvGetRows( data, &train_data, 0, ntrain_samples );

        // 2. train classifier
        int layer_sz[] = { data->cols, 100, 100, class_count };
        CvMat layer_sizes =
            cvMat( 1, (int)(sizeof(layer_sz)/sizeof(layer_sz[0])), CV_32S, layer_sz );
        mlp.create( &layer_sizes );
        printf( "Training the classifier (may take a few minutes)...\n");

#if 1
        int method = CvANN_MLP_TrainParams::BACKPROP;
        double method_param = 0.001;
        int max_iter = 300;
#else
        int method = CvANN_MLP_TrainParams::RPROP;
        double method_param = 0.1;
        int max_iter = 1000;
#endif

        mlp.train( &train_data, new_responses, 0, 0,
            CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER,max_iter,0.01),
                                  method, method_param));
        cvReleaseMat( &new_responses );
        printf("\n");
    }

    mlp_response = cvCreateMat( 1, class_count, CV_32F );

    // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        int best_class;
        CvMat sample;
        cvGetRow( data, &sample, i );
        CvPoint max_loc = {0,0};
        mlp.predict( &sample, mlp_response );
        cvMinMaxLoc( mlp_response, 0, 0, 0, &max_loc, 0 );
        best_class = max_loc.x + 'A';

        int r = fabs((double)best_class - responses->data.fl[i]) < FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all-ntrain_samples);
    train_hr /= (double)ntrain_samples;
    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    // Save classifier to file if needed
    if( filename_to_save )
        mlp.save( filename_to_save );

    cvReleaseMat( &mlp_response );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}

// Build the knearest classifier 
static
int build_knearest_classifier( char* data_filename, int K )
{
    const int var_count = 16;
    CvMat* data = 0;
    CvMat train_data;
    CvMat* responses;

    int ok = read_num_class_data( data_filename, 16, &data, &responses );
    int nsamples_all = 0, ntrain_samples = 0;
    //int i, j;
    //double /*train_hr = 0,*/ test_hr = 0;
    CvANN_MLP mlp;

    if( !ok )
    {
        printf( "Could not read the database %s\n", data_filename );
        return -1;
    }

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.8);

    // 1. unroll the responses
    printf( "Unrolling the responses...\n");
    cvGetRows( data, &train_data, 0, ntrain_samples );

    // 2. train classifier
    CvMat* train_resp = cvCreateMat( ntrain_samples, 1, CV_32FC1);
    for (int i = 0; i < ntrain_samples; i++)
        train_resp->data.fl[i] = responses->data.fl[i];
    CvKNearest knearest(&train_data, train_resp);

    CvMat* nearests = cvCreateMat( (nsamples_all - ntrain_samples), K, CV_32FC1);
    float* _sample = new float[var_count * (nsamples_all - ntrain_samples)];
    CvMat sample = cvMat( nsamples_all - ntrain_samples, 16, CV_32FC1, _sample );
    float* true_results = new float[nsamples_all - ntrain_samples];
    for (int j = ntrain_samples; j < nsamples_all; j++)
    {
        float *s = data->data.fl + j * var_count;

        for (int i = 0; i < var_count; i++)
        {
            sample.data.fl[(j - ntrain_samples) * var_count + i] = s[i];
        }
        true_results[j - ntrain_samples] = responses->data.fl[j];
    }
    CvMat *result = cvCreateMat(1, nsamples_all - ntrain_samples, CV_32FC1);
    knearest.find_nearest(&sample, K, result, 0, nearests, 0);
    int true_resp = 0;
    int accuracy = 0;
    for (int i = 0; i < nsamples_all - ntrain_samples; i++)
    {
        if (result->data.fl[i] == true_results[i])
            true_resp++;
        for(int k = 0; k < K; k++ )
        {
            if( nearests->data.fl[i * K + k] == true_results[i])
            accuracy++;
        }
    }

    printf("true_resp = %f%%\tavg accuracy = %f%%\n", (float)true_resp / (nsamples_all - ntrain_samples) * 100,
                                                      (float)accuracy / (nsamples_all - ntrain_samples) / K * 100);

    delete[] true_results;
    delete[] _sample;
    cvReleaseMat( &train_resp );
    cvReleaseMat( &nearests );
    cvReleaseMat( &result );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}

// Build the nbayes classifier
static
int build_nbayes_classifier( char* data_filename )
{
    const int var_count = 16;
    CvMat* data = 0;
    CvMat train_data;
    CvMat* responses;

    int ok = read_num_class_data( data_filename, 16, &data, &responses );
    int nsamples_all = 0, ntrain_samples = 0;
    //int i, j;
    //double /*train_hr = 0, */test_hr = 0;
    CvANN_MLP mlp;

    if( !ok )
    {
        printf( "Could not read the database %s\n", data_filename );
        return -1;
    }

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.5);

    // 1. unroll the responses
    printf( "Unrolling the responses...\n");
    cvGetRows( data, &train_data, 0, ntrain_samples );

    // 2. train classifier
    CvMat* train_resp = cvCreateMat( ntrain_samples, 1, CV_32FC1);
    for (int i = 0; i < ntrain_samples; i++)
        train_resp->data.fl[i] = responses->data.fl[i];
    CvNormalBayesClassifier nbayes(&train_data, train_resp);

    float* _sample = new float[var_count * (nsamples_all - ntrain_samples)];
    CvMat sample = cvMat( nsamples_all - ntrain_samples, 16, CV_32FC1, _sample );
    float* true_results = new float[nsamples_all - ntrain_samples];
    for (int j = ntrain_samples; j < nsamples_all; j++)
    {
        float *s = data->data.fl + j * var_count;

        for (int i = 0; i < var_count; i++)
        {
            sample.data.fl[(j - ntrain_samples) * var_count + i] = s[i];
        }
        true_results[j - ntrain_samples] = responses->data.fl[j];
    }
    CvMat *result = cvCreateMat(1, nsamples_all - ntrain_samples, CV_32FC1);
    nbayes.predict(&sample, result);
    int true_resp = 0;
    //int accuracy = 0;
    for (int i = 0; i < nsamples_all - ntrain_samples; i++)
    {
        if (result->data.fl[i] == true_results[i])
            true_resp++;
    }

    printf("true_resp = %f%%\n", (float)true_resp / (nsamples_all - ntrain_samples) * 100);

    delete[] true_results;
    delete[] _sample;
    cvReleaseMat( &train_resp );
    cvReleaseMat( &result );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}

// Build the svm classifier
static
int build_svm_classifier( char* data_filename, const char* filename_to_save, const char* filename_to_load )
{
    CvMat* data = 0;
    CvMat* responses = 0;
    CvMat* train_resp = 0;
    CvMat train_data;
    int nsamples_all = 0, ntrain_samples = 0;
    int var_count;
    CvSVM svm;

    int ok = read_num_class_data( data_filename, 16, &data, &responses );
    if( !ok )
    {
        printf( "Could not read the database %s\n", data_filename );
        return -1;
    }
    ////////// SVM parameters ///////////////////////////////
    CvSVMParams param;
    param.kernel_type=CvSVM::LINEAR;
    param.svm_type=CvSVM::C_SVC;
    param.C=1;
    ///////////////////////////////////////////////////////////

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.1);
    var_count = data->cols;

    // Create or load Random Trees classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        svm.load( filename_to_load );
        ntrain_samples = 0;
        if( svm.get_var_count() == 0 )
        {
            printf( "Could not read the classifier %s\n", filename_to_load );
            return -1;
        }
        printf( "The classifier %s is loaded.\n", filename_to_load );
    }
    else
    {
        // train classifier
        printf( "Training the classifier (may take a few minutes)...\n");
        cvGetRows( data, &train_data, 0, ntrain_samples );
        train_resp = cvCreateMat( ntrain_samples, 1, CV_32FC1);
        for (int i = 0; i < ntrain_samples; i++)
            train_resp->data.fl[i] = responses->data.fl[i];
        svm.train(&train_data, train_resp, 0, 0, param);
    }

    // classification
    std::vector<float> _sample(var_count * (nsamples_all - ntrain_samples));
    CvMat sample = cvMat( nsamples_all - ntrain_samples, 16, CV_32FC1, &_sample[0] );
    std::vector<float> true_results(nsamples_all - ntrain_samples);
    for (int j = ntrain_samples; j < nsamples_all; j++)
    {
        float *s = data->data.fl + j * var_count;

        for (int i = 0; i < var_count; i++)
        {
            sample.data.fl[(j - ntrain_samples) * var_count + i] = s[i];
        }
        true_results[j - ntrain_samples] = responses->data.fl[j];
    }
    CvMat *result = cvCreateMat(1, nsamples_all - ntrain_samples, CV_32FC1);

    printf("Classification (may take a few minutes)...\n");
    double t = (double)cvGetTickCount();
    svm.predict(&sample, result);
    t = (double)cvGetTickCount() - t;
    printf("Prediction type: %gms\n", t/(cvGetTickFrequency()*1000.));

    int true_resp = 0;
    for (int i = 0; i < nsamples_all - ntrain_samples; i++)
    {
        if (result->data.fl[i] == true_results[i])
            true_resp++;
    }

    printf("true_resp = %f%%\n", (float)true_resp / (nsamples_all - ntrain_samples) * 100);

    if( filename_to_save )
        svm.save( filename_to_save );

    cvReleaseMat( &train_resp );
    cvReleaseMat( &result );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}

// Helper function for running the benchmarks and creating the initial dataset that will be used to build the random forest

// Function that write the dataset on disk 
void write_in_dataset( string path, dataset_record * r){
	ofstream myfile (dataset_path,ios::app);
	/*
	cout<<"r->num_threads:"<<r->num_threads<<endl;
	cout<<"r->cpu_usage:"<<r->cpu_usage<<endl;
	cout<<"r->pid_cpu_usage:"<<r->pid_cpu_usage<<endl;
	cout<<"r->ram_usage:"<<r->ram_usage<<endl;
	cout<<"r->pid_ram_usage:"<<r->pid_ram_usage<<endl;
	cout<<"r->pid_page_faults:"<<r->pid_page_faults<<endl;
	cout<<"r->vm_usage:"<<r->vm_usage<<endl;
	cout<<"r->pid_vm_read_usage:"<<r->pid_vm_read_usage<<endl;
	cout<<"r->pid_vm_write_usage:"<<r->pid_vm_write_usage<<endl;
	cout<<"r->processes:"<<r->processes<<endl;
	cout<<"r->procs_running:"<<r->procs_running<<endl;
	cout<<"r->procs_blocked:"<<r->procs_blocked<<endl;
	cout<<"r->steal_time:"<<r->steal_time<<endl;
	cout<<"r->user:"<<r->user<<endl;
	cout<<"r->sys:"<<r->sys<<endl;
	cout<<"r->nice:"<<r->nice<<endl;
	cout<<"r->idle:"<<r->idle<<endl;
	cout<<"r->wait:"<<r->wait<<endl;
	cout<<"r->irq:"<<r->irq<<endl;
	*/
	//cout.setprecision(8);
    myfile <<r->num_threads<<","<<std::fixed<<std::setw(8)<<r->cpu_usage<<","<<r->pid_cpu_usage<<","<<r->lst_cpu_usage<<",";
	// First set
	//// included
	// not incl
	////myfile << std::fixed<<std::setw(8)<< r->ram_usage<<","<<r->pid_page_faults<<",";
	//myfile<<r->pid_ram_usage<<",";
	//myfile << std::fixed<<std::setw(8)<< r->vm_usage<<","<<r->pid_vm_read_usage<<","<<r->pid_vm_write_usage<<",";
	//myfile << std::fixed<<std::setw(8)<< r->vm_usage<<","<<r->pid_vm_read_usage<<","<<r->pid_vm_write_usage<<",";
	////myfile <<r->procs_blocked<<",";
	//myfile <<r->processes<<","<<r->procs_running<<","<<r->procs_blocked<<",";
    //myfile << std::fixed<<std::setw(8)<< r->user<<","<<r->sys<<","<<r->idle<<endl;
	////myfile << std::fixed<<std::setw(8)<<r->user<<","<<r->sys<<","<<r->idle<<","<<r->irq<<endl;
	//myfile << r->steal_time<<","<< std::fixed<<std::setw(8)<<r->user<<","<<r->sys<<","<<r->nice<<","<<r->idle<<","<<r->wait<<","<<r->irq<<","<<r->soft_irq<<",";
	//myfile << r->lst_stolen<<","<< r->lst_user<<","<<r->lst_sys<<","<<r->lst_nice<<","<<r->lst_idle<<","<<r->lst_wait<<","<<r->lst_irq<<","<<r->lst_soft_irq<<","<<r->lst_total<<",";
	//myfile <<  std::fixed<<std::setw(8) << r->lst_cpu_usage<<endl;
	
	// Second set, exp2-10 , 28
	
	myfile << std::fixed<<std::setw(8)<< r->ram_usage<<","<<r->pid_page_faults<<",";
	myfile<<r->pid_ram_usage<<",";
	myfile <<r->procs_blocked<<",";
	myfile <<r->processes<<","<<r->procs_running<<","<<r->procs_blocked<<",";
    myfile << std::fixed<<std::setw(8)<<r->user<<","<<r->sys<<","<<r->nice<<","<<r->idle<<",";
	myfile << std::fixed<<std::setw(8)<<r->wait<<","<<r->irq<<","<<r->soft_irq<<",";
	myfile << r->lst_stolen<<","<< r->lst_user<<","<<r->lst_sys<<","<<r->lst_nice<<","<<r->lst_idle<<","<<r->lst_wait<<","<<r->lst_irq<<","<<r->lst_soft_irq<<","<<r->lst_total<<",";
	myfile <<  std::fixed<<std::setw(8) << r->lst_cpu_usage<<endl;
  
    //Dataset #3
    /*
    myfile <<r->num_threads<<","<<std::fixed<<std::setw(8)<<r->pid_cpu_usage<<","<<r->lst_cpu_usage<<",";
	myfile <<r->nice<<","<<r->idle<<",";
	myfile << std::fixed<<std::setw(8)<<r->wait<<","<<r->irq<<","<<r->soft_irq<<",";
	myfile << std::fixed<<std::setw(8) << r->lst_cpu_usage<<endl;
    */
    //Datset 4
    /*
    myfile <<r->num_threads<<","<<std::fixed<<std::setw(8)<<r->pid_ram_usage<<","<<r->ram_usage<<","<<r->pid_cpu_usage<<","<<r->cpu_usage<<",";
    myfile <<r->procs_blocked<<",";
    myfile <<r->nice<<","<<r->idle<<",";
	myfile << std::fixed<<std::setw(8)<<r->user<<","<<r->sys<<","<<r->wait<<","<<r->irq<<","<<r->soft_irq<<",";
	myfile << std::fixed<<std::setw(8) << r->lst_cpu_usage<<endl;
	*/
	myfile.close();
}

// Function to find the min
int find_min(double array [], int size_array){
	double small = numeric_limits<double>::max();
	//puts("max:");
	cout.precision(8);
	//cout << "Max double =: " << fixed << small << endl;
	int index = -1;
	for (int i=0;i< size_array; i++){
		//cout << "In for "<< i<<":" << fixed << array[i] << endl;
		if(array[i]<small){
			//cout << "Found smaller "<<array[i]<<"than "<< fixed << small<< endl;
			small = array[i];
			index = i;
		}
	}
	//cout<<"At the end , the min index is:"<<index<<endl;
	if(index>=0) return index;
	else return -1;
}

// Helper Function that runs the linpack benchmarks,params: int num_tries, int n, int num_thr_start, int num_thr_end
void * linpack_benchmark(void * params){
	
	benchmark_param * linpack_params = static_cast<benchmark_param *> (params);
	
	int num_thr = linpack_params->num_thr_start;
	// time results
	int range_threads = linpack_params->num_thr_end - linpack_params->num_thr_start + 1;
	double results [range_threads];
	dataset_record records [range_threads];
	
	//cout<<"Linpack Benchmark"<<endl;
    //cout<<"===================================================="<<endl;
     
	
	// stats results
	
	double dy [linpack_params->n];
	double dx [linpack_params->n];
	double da = 3.14;
	
	for(int j =0; j<linpack_params->num_tries; j++){
		num_thr = 0;
		
		for(int k =0; k<range_threads; k++){
			// increment num threads
			if(num_thr<=linpack_params->num_thr_end){
				num_thr ++;
			}
			else {
				
				break;
			}
			// init dx
			for(int i=0; i < linpack_params->n; i++)
			  {
				  dx[i]=(double)i/5;
			  }
			  
			  // get stats
			  records[k].num_threads = num_thr;
			  records[k].processes = sysconf( _SC_NPROCESSORS_ONLN );
			  get_runtime_stats(&records[k]);
			 // run linpack benchmark
			 double start_time = omp_get_wtime();
			 #pragma omp parallel for num_threads(num_thr)
				 for(int i=0; i < linpack_params->n; i++)
				 {
					dy[i] = dy[i] + da * dx[i];
				 }
			double time = omp_get_wtime() - start_time;
		    cout<<"Elapsed time with "<<num_thr<<" threads:"<<time<<endl;
			results[k] = time;
			
			int micro = rand()%1000000 + 1;
			usleep(micro);
			//cout<<"Time elapsed:"<<time<<endl;
			
		}
		
		 int min_index = find_min(results, range_threads);
	     //cout<<"Min Time elapsed for run "<<j<<":"<<fixed<< results[min_index]<<endl; 
	     pthread_mutex_lock(&lock);
	     write_in_dataset("",&records[min_index]);
		 pthread_mutex_unlock(&lock);
		  
	}
     
}

// Gauss seidel benchmark, params : int num_tries, int n, int num_thr_start, int num_thr_end
void * gauss_seidel_benchmark(void * params){
	
	benchmark_param *gauss_seidel_params = static_cast<benchmark_param *> (params);
	
	int num_thr = gauss_seidel_params->num_thr_start;
	// time results
	int range_threads = gauss_seidel_params->num_thr_end - gauss_seidel_params->num_thr_start + 1;
	double results [range_threads];
	dataset_record records [range_threads];
	// stats results
	
	//cout<<"Gauss Seidel Benchmark"<<endl;
    //cout<<"===================================================="<<endl;
    
	int HEIGHT = gauss_seidel_params->n;
	int WIDTH = gauss_seidel_params->n;
	int DEPTH = gauss_seidel_params->n;
	
	for(int j =0; j<gauss_seidel_params->num_tries; j++){
		num_thr = 0;
		
		for(int k =0; k<range_threads; k++){
			// increment num threads
			if(num_thr<=gauss_seidel_params->num_thr_end){
				num_thr ++;
			}
			else {
				
				break;
			}
		
			// init 
			 srand (time(NULL));
			 vector<vector<vector<double> > > array3D;
			 
			 //std::cout << "CONFIGURATION\n";
			 //std::cout << "DIMENSIONS:"<<HEIGHT<<" "<<WIDTH<<" "<<DEPTH<<"\n";
			 //std::cout << "Parallel execution started ................\n";

			  // Set up sizes. (HEIGHT x WIDTH)
			  array3D.resize(HEIGHT);
			  for (int i = 0; i < HEIGHT; ++i) {
				array3D[i].resize(WIDTH);

				for (int j = 0; j < WIDTH; ++j)
				  array3D[i][j].resize(DEPTH);
			  }

				for (int i = 0; i < HEIGHT; ++i) {
					 for (int j = 0; j < WIDTH; ++j){
						for (int k = 0; k < DEPTH; ++k){
							double f = (double)rand() / RAND_MAX;
							array3D[i][j][k]= 1 + f * (1000 -1);
						}
					}
				}
				
				// get stats
				records[k].num_threads = num_thr;
				records[k].processes = sysconf( _SC_NPROCESSORS_ONLN );
				get_runtime_stats(&records[k]);
				/*
				std::cout<<"START INIT \n";
				
				for (int i = 0; i < HEIGHT; ++i) {
					 for (int j = 0; j < WIDTH; ++j){
						for (int k = 0; k < DEPTH; ++k){
							std::cout<<(array3D[i][j][k])<<" ";
						}
						std::cout<<"\n";
					}
					std::cout<<"\n";
				}
				std::cout<<"END INIT \n";
				* */
				
				int iter;
				double b= 1.0/6.0;
				int iterEnd=10;
				
				int ii,jj,kk,jStart,jEnd,threadID,numThreads;
	
			 // run linpack benchmark
			double start_time = omp_get_wtime();
			 
			 #pragma omp parallel shared(array3D,b) private (ii,jj,kk,jStart,jEnd,threadID) num_threads(num_thr)
			 {
				threadID=omp_get_thread_num();
				
				#pragma omp single 
				{
					numThreads=omp_get_num_threads();	
				}
				
				jStart=(threadID*((int)(WIDTH-1)/numThreads));
				
				
				jEnd=(jStart+(int)((WIDTH-1)/numThreads));
				
				
				
				for (int l=1;l<(DEPTH-1)+numThreads-1;l++){
					kk=l-threadID;
					if((kk>=1) and(kk<(DEPTH-1))) {
						
						for (jj=jStart+1;jj<jEnd;jj=jj+1){
							for(ii=1;ii<HEIGHT-1;ii++){
								
								array3D[ii][jj][kk] = b * ( array3D[ii-1][jj][kk] + array3D[ii+1][jj][kk] +
														array3D[ii][jj-1][kk] + array3D[ii][jj+1][kk] +
														array3D[ii][jj][kk-1] + array3D[ii][jj][kk+1]);
								
							}
							
						} 
					}
					#pragma omp barrier 
					
				}
				
			 }
			
			double time = omp_get_wtime() - start_time;
		    //cout<<"Elapsed time for "<<num_thr<<" threads:"<<time<<endl;
			results[k] = time;
			
			int micro = rand()%5500000 + 1;
			//usleep(micro);
			//for(int dd=0;dd<range_threads;dd++){
			//	cout<<"["+dd<<"] = "<<fixed<<results[dd]<<endl;
			//}
			//cout<<"Time elapsed:"<<time<<"wtf:"<<results[k]<<"Reuslt[0]="<<results[0]<<endl;
			
		}
		cout.precision(17);
		//cout<<"about to send results to min."<<endl;
		for(int d = 0;d<range_threads;d++){
			//cout<<d<<" : "<<fixed<<results[d];
		}
		 int min_index = find_min(results, range_threads);
	     //cout<<"Min Time elapsed for run "<<j<<":"<<fixed<< results[min_index]<<endl; 
	     pthread_mutex_lock(&lock);
	     write_in_dataset("",&records[min_index]);
		 pthread_mutex_unlock(&lock);
	}

}
// Helper Function that runs the dijkstra benchmarks,params: int num_tries, int n, int num_thr_start, int num_thr_end
void * dijkstra_benchmark(void * params){
	
	benchmark_param * dijkstra_params = static_cast<benchmark_param *> (params);
	
	int num_thr = dijkstra_params->num_thr_start;
	// time results
	int range_threads = dijkstra_params->num_thr_end - dijkstra_params->num_thr_start + 1;
	double results [range_threads];
	dataset_record records [range_threads];
	
	//cout<<"Dijkstra Benchmark"<<endl;
    //cout<<"===================================================="<<endl;
     
	
	// stats results
		
	for(int j =0; j<dijkstra_params->num_tries; j++){
		num_thr = 0;
		for(int k =0; k<range_threads; k++){
			// increment num threads
			if(num_thr<=dijkstra_params->num_thr_end){
				num_thr ++;
			}
			else {
				break;
			}
		
			// set n?
			  
			  // get stats
			  records[k].num_threads = num_thr;
			  records[k].processes = sysconf( _SC_NPROCESSORS_ONLN );
			  get_runtime_stats(&records[k]);
			  
			 // run dijkstra benchmark
			 double start_time = omp_get_wtime();
			
			//omp_set_num_threads(num_thr);
			run_dijkstra(num_thr);
			 
			double time = omp_get_wtime() - start_time;
		    //cout<<"Elapsed time for "<<num_thr<<" threads:"<<time<<endl;
			results[k] = time;
			
			//int micro = rand()%10000 + 1;
			//usleep(micro);
			//cout<<"Time elapsed:"<<time<<endl;
			
		}
		
		 int min_index = find_min(results, range_threads);
	     //cout<<"Min Time elapsed for run "<<j<<":"<<fixed<< results[min_index]<<endl; 
	     pthread_mutex_lock(&lock);
	     write_in_dataset("",&records[min_index]);
		 pthread_mutex_unlock(&lock);
		  
	}
     
}

// Helper Function that runs the multitask benchmarks,params: int num_tries, int n, int num_thr_start, int num_thr_end
void * multitask_benchmark(void * params){
	
	benchmark_param * multitask_params = static_cast<benchmark_param *> (params);
	
	int num_thr = multitask_params->num_thr_start;
	// time results
	int range_threads = multitask_params->num_thr_end - multitask_params->num_thr_start + 1;
	double results [range_threads];
	dataset_record records [range_threads];
	
	//cout<<"<Multitask Benchmark"<<endl;
    //cout<<"===================================================="<<endl;
     
	
	// stats results
		
	for(int j =0; j<multitask_params->num_tries; j++){
		num_thr = 0;
		
		for(int k =0; k<range_threads; k++){
			// increment num threads
			if(num_thr<=multitask_params->num_thr_end){
				num_thr ++;
			}
			else {
				
				break;
			}
		
			// set n?
			  
			  // get stats
			  records[k].num_threads = num_thr;
			  records[k].processes = sysconf( _SC_NPROCESSORS_ONLN );
			  get_runtime_stats(&records[k]);
			  
			 // run linpack benchmark
			 double start_time = omp_get_wtime();
			
			//omp_set_num_threads(num_thr);
			run_multitask(num_thr);
			 
			double time = omp_get_wtime() - start_time;
		    //cout<<"Elapsed time with "<<num_thr<<" threads: "<<time<<endl;
			results[k] = time;
			
			//int micro = rand()%10000 + 1;
			//usleep(micro);
			//cout<<"Time elapsed:"<<time<<endl;
			
		}
		
		 int min_index = find_min(results, range_threads);
	     //cout<<"Min Time elapsed for run "<<j<<":"<<fixed<< results[min_index]<<endl; 
	     pthread_mutex_lock(&lock);
	     write_in_dataset("",&records[min_index]);
		 pthread_mutex_unlock(&lock);
		  
	}
     
}

// Helper Function that runs the poisson benchmarks,params: int num_tries, int n, int num_thr_start, int num_thr_end
void * poisson_benchmark(void * params){
	
	benchmark_param * poisson_params = static_cast<benchmark_param *> (params);
	
	int num_thr = poisson_params->num_thr_start;
	// time results
	int range_threads = poisson_params->num_thr_end - poisson_params->num_thr_start + 1;
	double results [range_threads];
	dataset_record records [range_threads];
	
	//cout<<"<Poisson Benchmark"<<endl;
    //cout<<"===================================================="<<endl;
     
	
	// stats results
		
	for(int j =0; j<poisson_params->num_tries; j++){
		num_thr = 0;
		
		for(int k =0; k<range_threads; k++){
			// increment num threads
			if(num_thr<=poisson_params->num_thr_end){
				num_thr ++;
			}
			else {
				
				break;
			}
		
			// set n?
			  
			  // get stats
			  records[k].num_threads = num_thr;
			  records[k].processes = sysconf( _SC_NPROCESSORS_ONLN );
			  get_runtime_stats(&records[k]);
			  
			 // run linpack benchmark
			 double start_time = omp_get_wtime();
			
			//omp_set_num_threads(num_thr);
			run_poisson(num_thr);
			
			double time = omp_get_wtime() - start_time;
		    //cout<<"Elapsed time with "<<num_thr<<" threads:"<<time<<endl;
			results[k] = time;
	
		    int micro = rand()%30000000 + 10000;
			usleep(micro);

			//cout<<"Time elapsed:"<<time<<endl;
			
		}
		
		 int min_index = find_min(results, range_threads);
	     //cout<<"Min Time elapsed for run "<<j<<":"<<fixed<< results[min_index]<<endl; 
	     pthread_mutex_lock(&lock);
	     write_in_dataset("",&records[min_index]);
		 pthread_mutex_unlock(&lock);
		  
	}
     
}


void * circuit_benchmark(void * params){
	benchmark_param * circuit_params = static_cast<benchmark_param *> (params);
	
	int num_thr = circuit_params->num_thr_start;
	// time results
	int range_threads = circuit_params->num_thr_end - circuit_params->num_thr_start + 1;
	double results [range_threads];
	dataset_record records [range_threads];
	
	//cout<<"<Circuit Benchmark"<<endl;
    //cout<<"===================================================="<<endl;
     
	
	// stats results
		
	for(int j =0; j<circuit_params->num_tries; j++){
		num_thr = 0;
		
		for(int k =0; k<range_threads; k++){
			// increment num threads
			if(num_thr<=circuit_params->num_thr_end){
				num_thr ++;
			}
			else {
				
				break;
			}
		
			// set n?
			  
			  // get stats
			  records[k].num_threads = num_thr;
			  records[k].processes = sysconf( _SC_NPROCESSORS_ONLN );
			  get_runtime_stats(&records[k]);
			  
			 // run linpack benchmark
			 double start_time = omp_get_wtime();
			//omp_set_num_threads(num_thr);
			run_circuit(num_thr);
			 
			double time = omp_get_wtime() - start_time;
			//cout<<"Running with "<<num_thr<<" threads: "<<time<<endl;
			results[k] = time;
			
			int micro = rand()%3000000 + 1000;
			usleep(micro);
			//cout<<"Time elapsed:"<<time<<endl;
			
		}
		
		 int min_index = find_min(results, range_threads);
	     //cout<<"Min Time elapsed for run "<<j<<":"<<fixed<< results[min_index]<<endl; 
	     pthread_mutex_lock(&lock);
	     write_in_dataset("",&records[min_index]);
		 pthread_mutex_unlock(&lock);
		  
	}

}

// Function that runs all the benchmarks, and creates the forest
void create_forest(){
	
	//char* filename_to_save = 0;
    //char* filename_to_load = 0;
    char default_data_filename[] = dataset_path;
    char default_data_filename1[] = forest_path;
    char* data_filename = default_data_filename;
    char * filename_to_save = default_data_filename1;
    char default_data_filename2[] = forest_path;
    // Uncomment if you want to load tree
    forest_created=true;
    char * filename_to_load = default_data_filename2;
    int method = 0;
   
	cout<<"About to start running the benchmarks and create the random trees."<<endl;
	
	if(!forest_created) {
	// Create threads to run the benchamarks
	pthread_t linpack_thread, gauss_seidel_thread, dijkstra_thread, multitask_thread, poisson_thread;
	pthread_t linpack_thread1, gauss_seidel_thread1, dijkstra_thread1, multitask_thread1, poisson_thread1;
	pthread_t circuit_thread, circuit_thread1;
	// error codes
	int  iret1, iret2, iret3, iret4, iret5;
	int iret6, iret7, iret8, iret9, iret10;
	int iret11, iret12;
	
	int max_num_thr = omp_get_max_threads();
	
	benchmark_param linpack_param;
	linpack_param.n = 900000;
	linpack_param.num_tries = 25;
	linpack_param.num_thr_start = 1;
	linpack_param.num_thr_end = max_num_thr;

	benchmark_param gaus_seidel_param;
	gaus_seidel_param.n = 500;
	gaus_seidel_param.num_tries = 10;
	gaus_seidel_param.num_thr_start = 1;
	gaus_seidel_param.num_thr_end = max_num_thr;
	
	benchmark_param dijkstra_param;
	dijkstra_param.n = 100; // not used, fixed 600
	dijkstra_param.num_tries = 25;
	dijkstra_param.num_thr_start = 1;
	dijkstra_param.num_thr_end = max_num_thr;
	
	benchmark_param multitask_param;
	multitask_param.n = 100; // not used, fixed 2* 10000
	multitask_param.num_tries = 30;
	multitask_param.num_thr_start = 1;
	multitask_param.num_thr_end = max_num_thr;
	
	benchmark_param poisson_param;
	poisson_param.n = 200; // uses 100
	poisson_param.num_tries = 62;
	poisson_param.num_thr_start = 1;
	poisson_param.num_thr_end = max_num_thr;
	
	benchmark_param circuit_param;
	circuit_param.n = 200; // uses 100
	circuit_param.num_tries = 63;
	circuit_param.num_thr_start = 1;
	circuit_param.num_thr_end = max_num_thr;
	
	if(pthread_mutex_init(&lock, NULL)!=0){
		cout<<"An error occured with pthreads."<<endl;
		return;
	}
	
	//linpack_benchmark(&linpack_param);
	//gauss_seidel_benchmark(&gaus_seidel_param);
	//dijkstra_benchmark(&dijkstra_param);
	//multitask_benchmark(&multitask_param);
	//poisson_benchmark(&poisson_param);
	//circuit_benchmark(&circuit_param);
	
	/*
	iret1 = pthread_create(&linpack_thread, NULL, linpack_benchmark, (void *)&linpack_param);
	if (iret1){
      cout<<"ERROR; return code from pthread_create() is "<<iret1<<endl;
         exit(-1);
    }
    
   
	iret2 = pthread_create(&gauss_seidel_thread, NULL, gauss_seidel_benchmark, (void *)&gaus_seidel_param);
	if (iret2){
      cout<<"ERROR; return code from pthread_create() is "<<iret2<<endl;
         exit(-1);
    }
    */
    /*
	iret3 = pthread_create(&dijkstra_thread, NULL, dijkstra_benchmark, (void *)&dijkstra_param);
	if (iret3){
      cout<<"ERROR; return code from pthread_create() is "<<iret3<<endl;
         exit(-1);
    }
    * */
    
    int micro = 1000000;
	usleep(micro);
    
    iret4 = pthread_create(&multitask_thread, NULL, multitask_benchmark, (void *)&multitask_param);
	if (iret4){
      cout<<"ERROR; return code from pthread_create() is "<<iret4<<endl;
         exit(-1);
    }
    
    //usleep(micro);
   //int micro = 30000000;
    usleep(micro);
    
    iret5 = pthread_create(&poisson_thread, NULL, poisson_benchmark, (void *)&poisson_param);
	if (iret5){
      cout<<"ERROR; return code from pthread_create() is "<<iret5<<endl;
         exit(-1);
    }
    
  // int micro = 15000000;
    usleep(micro);
    
    iret11 = pthread_create(&circuit_thread, NULL, circuit_benchmark, (void *)&circuit_param);
	if (iret11){
      cout<<"ERROR; return code from pthread_create() is "<<iret11<<endl;
         exit(-1);
    }
    
    
     
    /*
    iret6 = pthread_create(&linpack_thread1, NULL, linpack_benchmark, (void *)&linpack_param);
	if (iret6){
      cout<<"ERROR; return code from pthread_create() is "<<iret6<<endl;
         exit(-1);
    }
   
	iret7 = pthread_create(&gauss_seidel_thread1, NULL, gauss_seidel_benchmark, (void *)&gaus_seidel_param);
	if (iret7){
      cout<<"ERROR; return code from pthread_create() is "<<iret7<<endl;
         exit(-1);
    }
    */
    /*
    usleep(micro);
	iret8 = pthread_create(&dijkstra_thread1, NULL, dijkstra_benchmark, (void *)&dijkstra_param);
	if (iret8){
      cout<<"ERROR; return code from pthread_create() is "<<iret8<<endl;
         exit(-1);
    }
    usleep(micro);
    */
    iret9 = pthread_create(&multitask_thread1, NULL, multitask_benchmark, (void *)&multitask_param);
	if (iret9){
      cout<<"ERROR; return code from pthread_create() is "<<iret9<<endl;
         exit(-1);
    }
   
    usleep(micro);
    iret10 = pthread_create(&poisson_thread1, NULL, poisson_benchmark, (void *)&poisson_param);
	if (iret10){
      cout<<"ERROR; return code from pthread_create() is "<<iret10<<endl;
         exit(-1);
    }
    usleep(micro);
    iret12 = pthread_create(&circuit_thread1, NULL, circuit_benchmark, (void *)&circuit_param);
	if (iret12){
      cout<<"ERROR; return code from pthread_create() is "<<iret12<<endl;
         exit(-1);
    } 
    
	// Join threads
	//pthread_join( linpack_thread, NULL);
    //pthread_join( gauss_seidel_thread, NULL);
     // pthread_join( dijkstra_thread, NULL);
      pthread_join( multitask_thread, NULL);
      pthread_join( poisson_thread, NULL);
      pthread_join( circuit_thread, NULL);
    //pthread_join( linpack_thread1, NULL);
    //pthread_join( gauss_seidel_thread1, NULL);
    
   // pthread_join( dijkstra_thread1, NULL);
    pthread_join( multitask_thread1, NULL);
    
    pthread_join( poisson_thread1, NULL);
    pthread_join( circuit_thread1, NULL);
    pthread_mutex_destroy(&lock);
	}
	//linpack_benchmark(2,100000, 1 , 8);
	//gauss_seidel_benchmark(2,99,1,8);
	//forest_created = false;
	build_rtrees_classifier( data_filename, filename_to_save, filename_to_load );	
	forest_created = true;
	
	}

// Function that interposes and determines the number of threads

    void omp_set_num_threads(int num_threads){
	
	puts("Measuring statistics");
	puts("==========================================================");
	
	void (*new_set_num_threads) (int num_threads);
	
	cout<<"Your original param was:" << num_threads << endl;
	
	new_set_num_threads = (void (*)(int)) dlsym(RTLD_NEXT, "omp_set_num_threads");

	// predict num_threads usnig forest , then invoke it with the resulting number
	//new_set_num_threads(10);
	if(!forest_created){
		cout<<"create forest now"<<endl;
		create_forest();
	}
	
	dataset_record current_state;
    get_runtime_stats(&current_state);
	double pred_res = predict_for_sample(&current_state);
	pred_res = pred_res - 48;
	
    cout<<"Predict result = "<<pred_res<<endl;
    cout<<"The benchmarks have finished running."<<endl;
    ofstream myfile;
    myfile.open (prediction_log,ios::app);
	myfile<<pred_res<<endl; 
	
	return new_set_num_threads(pred_res);
	
}





