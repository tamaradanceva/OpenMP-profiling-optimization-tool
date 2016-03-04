#include <iostream>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include "omp.h"
#include <fstream>
# include <cstdlib>
# include <iomanip>
# include <ctime>
# include <cmath>
#include <vector>

using namespace std;

struct benchmark_param{
	int n;
	int num_tries;
	int num_thr_start;
	int num_thr_end;
};

/*
 * Functions for the fast fourier transform
 * */
using namespace std;

int run_fft (int n_thr, int num_try  );
void ccopy ( int n, double x[], double y[] );
void cfft2 ( int n, double x[], double y[], double w[], double sgn );
void cffti ( int n, double w[] );
double ggl ( double *ds );
void step ( int n, int mj, double a[], double b[], double c[], double d[], 
  double w[], double sgn );
  void timestamp ( );

//****************************************************************************80

int run_fft (int n_thr, int num_try )

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for FFT_OPENMP.
//
//  Discussion:
//
//    The complex data in an N vector is stored as pairs of values in a
//    real vector of length 2*N.
//
//  Modified:
//
//    17 April 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
//
{
  double error;
  int first;
  double flops;
  double fnm1;
  int i;
  int icase;
  int it;
  int ln2;
  int ln2_max = 25;
  double mflops;
  int n;
  int nits = 10000;
  static double seed;
  double sgn;
  double *w;
  double wtime;
  double *x;
  double *y;
  double *z;
  double z0;
  double z1;

  timestamp ( );
  /*
  cout << "\n";
  cout << "FFT_OPENMP\n";
  cout << "  C++/OpenMP version\n";
  cout << "\n";
  cout << "  Demonstrate an implementation of the Fast Fourier Transform\n";
  cout << "  of a complex data vector, using OpenMP for parallel execution.\n";

  cout << "\n";
  cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
  cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";
*/
//
//  Prepare for tests.
//
/*
  cout << "\n";
  cout << "  Accuracy check:\n";
  cout << "\n";
  cout << "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n";
  cout << "\n";
  cout << "             N      NITS    Error         Time          Time/Call     MFLOPS\n";
  cout << "\n";
*/
  seed  = 331.0;
  n = 1;
//
//  LN2 is the log base 2 of N.  Each increase of LN2 doubles N.
//
  for ( ln2 = 1; ln2 <= 25; ln2++ )
  {
	
    n = 2 * n;
//
//  Allocate storage for the complex arrays W, X, Y, Z.  
//
//  We handle the complex arithmetic,
//  and store a complex number as a pair of doubles, a complex vector as a doubly
//  dimensioned array whose second dimension is 2. 
//
    w = new double[  n];
    x = new double[2*n];
    y = new double[2*n];
    z = new double[2*n];

    first = 1;

    for ( icase = 0; icase < 2; icase++ )
    {

      if ( first )
      {
        for ( i = 0; i < 2 * n; i = i + 2 )
        {
          z0 = ggl ( &seed );
          z1 = ggl ( &seed );
          x[i] = z0;
          z[i] = z0;
          x[i+1] = z1;
          z[i+1] = z1;
        }
      } 
      else
      {
# pragma omp parallel \
    shared ( n, x, z ) \
    private ( i, z0, z1 )

# pragma omp for nowait
        for ( i = 0; i < 2 * n; i = i + 2 )
        {
          z0 = 0.0;
          z1 = 0.0;
          x[i] = z0;
          z[i] = z0;
          x[i+1] = z1;
          z[i+1] = z1;
        }
      }
//
//  Initialize the sine and cosine tables.
//
      cffti ( n, w );
//
//  Transform forward, back 
//
      if ( first )
      {
        sgn = + 1.0;
        cfft2 ( n, x, y, w, sgn );
        sgn = - 1.0;
        cfft2 ( n, y, x, w, sgn );
// 
//  Results should be same as initial multiplied by N.
//
        fnm1 = 1.0 / ( double ) n;
        error = 0.0;
        for ( i = 0; i < 2 * n; i = i + 2 )
        {
          error = error 
          + pow ( z[i]   - fnm1 * x[i], 2 )
          + pow ( z[i+1] - fnm1 * x[i+1], 2 );
        }
        error = sqrt ( fnm1 * error );
       /*
        cout << "  " << setw(12) << n
             << "  " << setw(8) << nits
             << "  " << setw(12) << error;
       */
        first = 0;
        /*
        wtime = omp_get_wtime ( ) - wtime;
		  ofstream myfile;
		  myfile.open ("experiments/alg/pred_exp1_fft_forward.txt",ios::app);
		  //myfile<<num_try<<" , "<<num_thrs<<" , "<<wtime<<endl;
		  myfile<<wtime<<endl;
		  * */
      }
      else
      {
        wtime = omp_get_wtime ( );
        for ( it = 0; it < nits; it++ )
        {
          sgn = + 1.0;
          cfft2 ( n, x, y, w, sgn );
          sgn = - 1.0;
          cfft2 ( n, y, x, w, sgn );
        }
        wtime = omp_get_wtime ( ) - wtime;

        flops = ( double ) 2 * ( double ) nits 
          * ( ( double ) 5 * ( double ) n * ( double ) ln2 );

        mflops = flops / 1.0E+06 / wtime;
		/*
        cout << "  " << setw(12) << ctime
             << "  " << setw(12) << wtime / ( double ) ( 2 * nits )
             << "  " << setw(12) << mflops << "\n";
         */
         int num_threads;
         #pragma omp parallel 
         {
			num_threads = omp_get_num_threads();
		 }
         ofstream myfile;
		  if(ln2 == 25){
		  myfile.open ("experiments/alg/fft_alone.txt",ios::app);
		  myfile<<num_try<<" , "<<n<<" , "<<num_threads<<" , "<<wtime / ( double ) ( 2 * nits )<<endl;
		 // myfile<<wtime<<endl;
		}
        wtime = omp_get_wtime ( ) - wtime;
		  
		
      }
    }
    if ( ( ln2 % 4 ) == 0 ) 
    {
      nits = nits / 10;
    }
    if ( nits < 1 ) 
    {
      nits = 1;
    }
    delete [] w;
    delete [] x;
    delete [] y;
    delete [] z;
  }
//
//  Terminate.
//
/*
  cout << "\n";
  cout << "FFT_OPENMP:\n";
  cout << "  Normal end of execution.\n";
  cout << "\n";
  timestamp ( );
  * */


  return 0;
}
//****************************************************************************80

void ccopy ( int n, double x[], double y[] )

//****************************************************************************80
//
//  Purpose:
//
//    CCOPY copies a complex vector.
//
//  Discussion:
//
//    The "complex" vector A[N] is actually stored as a double vector B[2*N].
//
//    The "complex" vector entry A[I] is stored as:
//
//      B[I*2+0], the real part,
//      B[I*2+1], the imaginary part.
//
//  Modified:
//
//    20 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
//
//  Parameters:
//
//    Input, int N, the length of the "complex" array.
//
//    Input, double X[2*N], the array to be copied.
//
//    Output, double Y[2*N], a copy of X.
//
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    y[i*2+0] = x[i*2+0];
    y[i*2+1] = x[i*2+1];
   }
  return;
}
//****************************************************************************80

void cfft2 ( int n, double x[], double y[], double w[], double sgn )

//****************************************************************************80
//
//  Purpose:
//
//    CFFT2 performs a complex Fast Fourier Transform.
//
//  Modified:
//
//    20 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
//
//  Parameters:
//
//    Input, int N, the size of the array to be transformed.
//
//    Input/output, double X[2*N], the data to be transformed.  
//    On output, the contents of X have been overwritten by work information.
//
//    Output, double Y[2*N], the forward or backward FFT of X.
//
//    Input, double W[N], a table of sines and cosines.
//
//    Input, double SGN, is +1 for a "forward" FFT and -1 for a "backward" FFT.
//
{
  int j;
  int m;
  int mj;
  int tgle;

   m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) );
   mj   = 1;
//
//  Toggling switch for work array.
//
  tgle = 1;
  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );

  if ( n == 2 )
  {
    return;
  }

  for ( j = 0; j < m - 2; j++ )
  {
    mj = mj * 2;
    if ( tgle )
    {
      step ( n, mj, &y[0*2+0], &y[(n/2)*2+0], &x[0*2+0], &x[mj*2+0], w, sgn );
      tgle = 0;
    }
    else
    {
      step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
      tgle = 1;
    }
  }
//
//  Last pass thru data: move y to x if needed 
//
  if ( tgle ) 
  {
    ccopy ( n, y, x );
  }

  mj = n / 2;
  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );

  return;
}
//****************************************************************************80

void cffti ( int n, double w[] )

//****************************************************************************80
//
//  Purpose:
//
//    CFFTI sets up sine and cosine tables needed for the FFT calculation.
//
//  Modified:
//
//    20 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
//
//  Parameters:
//
//    Input, int N, the size of the array to be transformed.
//
//    Output, double W[N], a table of sines and cosines.
//
{
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ( ( double ) n );

# pragma omp parallel \
    shared ( aw, n, w ) \
    private ( arg, i )

# pragma omp for nowait

  for ( i = 0; i < n2; i++ )
  {
    arg = aw * ( ( double ) i );
    w[i*2+0] = cos ( arg );
    w[i*2+1] = sin ( arg );
  }
  return;
}
//****************************************************************************80

double ggl ( double *seed )

//****************************************************************************80
//
//  Purpose:
//
//    GGL generates uniformly distributed pseudorandom numbers. 
//
//  Modified:
//
//    20 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen, M Troyer, I Vattulainen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
//
//  Parameters:
//
//    Input/output, double *SEED, used as a seed for the sequence.
//
//    Output, double GGL, the next pseudorandom value.
//
{
  double d2 = 0.2147483647e10;
  double t;
  double value;

  t = ( double ) *seed;
  t = fmod ( 16807.0 * t, d2 );
  *seed = ( double ) t;
  value = ( double ) ( ( t - 1.0 ) / ( d2 - 1.0 ) );

  return value;
}
//****************************************************************************80

void step ( int n, int mj, double a[], double b[], double c[],
  double d[], double w[], double sgn )

//****************************************************************************80
//
//  Purpose:
//
//    STEP carries out one step of the workspace version of CFFT2.
//
//  Modified:
//
//    20 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
//
{
  double ambr;
  double ambu;
  int j;
  int ja;
  int jb;
  int jc;
  int jd;
  int jw;
  int k;
  int lj;
  int mj2;
  double wjw[2];

  mj2 = 2 * mj;
  lj = n / mj2;

# pragma omp parallel \
    shared ( a, b, c, d, lj, mj, mj2, sgn, w ) \
    private ( ambr, ambu, j, ja, jb, jc, jd, jw, k, wjw )

# pragma omp for nowait

  for ( j = 0; j < lj; j++ )
  {
    jw = j * mj;
    ja  = jw;
    jb  = ja;
    jc  = j * mj2;
    jd  = jc;

    wjw[0] = w[jw*2+0]; 
    wjw[1] = w[jw*2+1];

    if ( sgn < 0.0 ) 
    {
      wjw[1] = - wjw[1];
    }

    for ( k = 0; k < mj; k++ )
    {
      c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];
      c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];

      ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0];
      ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1];

      d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;
      d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;
    }
  }
  return;
}

/*
 * Funcitons for the the multitask benchmark
 * */
 
int run_multitask(int num_thrs, int num_try);
int *prime_table ( int prime_num );
double *sine_table ( int sine_num );

int run_multitask (int num_thrs, int num_try)
{
  int prime_num;
  int *primes;
  int sine_num;
  double *sines;
  double wtime;
  double wtime1;
  double wtime2;
  
  prime_num = 18500;
  sine_num = 18500;
  # pragma omp parallel
  {
  num_thrs = omp_get_num_threads();
  }
 // num_thrs=1;
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
  ofstream myfile;
  myfile.open ("experiments/alg/pred_exp21_mul.txt",ios::app);
  myfile<<num_try<<" , "<<num_thrs<<" , "<<wtime<<endl;
  free ( primes );
  free ( sines );
//
//  Terminate.
//

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


// Helper Function that runs the multitask benchmarks,params: int num_tries, int n, int num_thr_start, int num_thr_end
void multitask_benchmark(benchmark_param * multitask_params){
	
	//benchmark_param * multitask_params = static_cast<benchmark_param *> (params);
	
	int num_thr = multitask_params->num_thr_start;
	// time results
	int range_threads = multitask_params->num_thr_end - multitask_params->num_thr_start + 1;
	double results [range_threads];
	
	//cout<<"<Multitask Benchmark"<<endl;
    //cout<<"===================================================="<<endl;
     
	
	// stats results
		
	//for(int j =0; j<multitask_params->num_tries; j++){
		num_thr = 0;
		
		for(int k =0; k<range_threads; k++){
			// increment num threads
			if(num_thr<=multitask_params->num_thr_end){
				num_thr ++;
			}
			else {
				
				break;
			}
	
			 // run linpack benchmark
			 double start_time = omp_get_wtime();
			
			//omp_set_num_threads(num_thr);
			run_multitask(num_thr,1);
			 
			double time = omp_get_wtime() - start_time;
		    cout<<"Elapsed time with "<<num_thr<<" threads: "<<time<<endl;
			results[k] = time;
			
			//int micro = rand()%10000 + 1;
			//usleep(micro);
			//cout<<"Time elapsed:"<<time<<endl;
			
		}
		  
	//}
     
}


# define NV 1440

int run_dijkstra(int num_thrs);
int *dijkstra_distance ( int ohd[NV][NV], int num_thrs);
void find_nearest ( int s, int e, int mind[NV], bool connected[NV], int *d, 
  int *v );
void init ( int ohd[NV][NV] );
void timestamp ( );
void update_mind ( int s, int e, int mv, bool connected[NV], int ohd[NV][NV], 
  int mind[NV] );
  
// Helper Function that runs the dijkstra benchmarks,params: int num_tries, int n, int num_thr_start, int num_thr_end
void dijkstra_benchmark(benchmark_param * dijkstra_params){
	
	//benchmark_param * dijkstra_params = static_cast<benchmark_param *> (params);
	
	int num_thr = dijkstra_params->num_thr_start;
	// time results
	int range_threads = dijkstra_params->num_thr_end - dijkstra_params->num_thr_start + 1;
	double results [range_threads];
	
	//cout<<"Dijkstra Benchmark"<<endl;
    //cout<<"===================================================="<<endl;
     
	
	// stats results
		
//	for(int j =0; j<dijkstra_params->num_tries; j++){
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
			  
			 // run dijkstra benchmark
			 double start_time = omp_get_wtime();
			
			//omp_set_num_threads(num_thr);
			run_dijkstra(num_thr);
			 
			double time = omp_get_wtime() - start_time;
		    //cout<<"Elapsed time for "<<num_thr<<" threads:"<<time<<endl;
			results[k] = time;
			
			//int micro = rand()%10000 + 1;
			//usleep(micro);
			cout<<"Time elapsed for "<<num_thr<<":"<<time<<endl;
			
		}
		
		  
//	}
     
}


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


// Gauss seidel benchmark, params : int num_tries, int n, int num_thr_start, int num_thr_end
void gauss_seidel_benchmark(benchmark_param * gauss_seidel_params){
	
	//benchmark_param *gauss_seidel_params = static_cast<benchmark_param *> (params);
	
	int num_thr = gauss_seidel_params->num_thr_start;
	// time results
	int range_threads = gauss_seidel_params->num_thr_end - gauss_seidel_params->num_thr_start + 1;
	double results [range_threads];
	// stats results
	
	//cout<<"Gauss Seidel Benchmark"<<endl;
    //cout<<"===================================================="<<endl;
    
	int HEIGHT = gauss_seidel_params->n;
	int WIDTH = gauss_seidel_params->n;
	int DEPTH = gauss_seidel_params->n;
	
	//for(int j =0; j<gauss_seidel_params->num_tries; j++){
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
		    cout<<"Elapsed time for "<<num_thr<<" threads:"<<time<<endl;
			results[k] = time;
			
			int micro = rand()%5500 + 1;
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
	//}

}

// Helper Function that runs the linpack benchmarks,params: int num_tries, int n, int num_thr_start, int num_thr_end
void linpack_benchmark(benchmark_param * linpack_params){
	
	
	int num_thr = linpack_params->num_thr_start;
	// time results
	int range_threads = linpack_params->num_thr_end - linpack_params->num_thr_start + 1;
	double results [range_threads];
	
	//cout<<"Linpack Benchmark"<<endl;
    //cout<<"===================================================="<<endl;
     
	
	// stats results
	
	double dy [linpack_params->n];
	double dx [linpack_params->n];
	double da = 3.14;
	
	//for(int j =0; j<linpack_params->num_tries; j++){
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
			
			int micro = rand()%10000 + 1;
			usleep(micro);
			//cout<<"Time elapsed:"<<time<<endl;
			
		}
	
		  
	//}
     
}


int run_sgefa ( int num_try);
void test01 ( int n );
void test02 ( int n , int num_try);
void test03 ( int n );

int isamax ( int n, float x[], int incx );
void matgen ( int lda, int n, float a[], float x[], float b[] );
void msaxpy ( int nr, int nc, float a[], int n, float x[], float y[] );
void msaxpy2 ( int nr, int nc, float a[], int n, float x[], float y[] );
int msgefa ( float a[], int lda, int n, int ipvt[] );
int msgefa2 ( float a[], int lda, int n, int ipvt[] );
void saxpy ( int n, float a, float x[], int incx, float y[], int incy );
float sdot ( int n, float x[], int incx, float y[], int incy );
int sgefa ( float a[], int lda, int n, int ipvt[] );
void sgesl ( float a[], int lda, int n, int ipvt[], float b[], int job );
void sscal ( int n, float a, float x[], int incx );
void sswap ( int n, float x[], int incx, float y[], int incy );
void timestamp ( );

//****************************************************************************80

int run_sgefa (int num_try )

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for the SGEFA_OPENMP test program.
//
//  Discussion:
//
//    We want to compare methods of solving the linear system A*x=b.
//
//    The first way uses the standard sequential algorithm "SGEFA".
//
//    The second way uses a variant of SGEFA that has been modified to
//    take advantage of OpenMP.
//
//    The third way reruns the variant code, but with OpenMP turned off.
//
//  Modified:
//
//    07 April 2008
//
//  Author:
//
//    John Burkardt
//
{
  int n;

  timestamp ( );

  cout << "\n";
  cout << "SGEFA_OPENMP\n";
  cout << "  C++/OpenMP version\n";

  cout << "\n";
  cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
  cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";

  cout << "\n";
  cout << " Algorithm        Mode          N    Error       Time\n";

  cout << "\n";
  n = 10;
  //test01 ( n );
  test02 ( n, num_try );
  //test03 ( n );

  cout << "\n";
  n = 100;
  //test01 ( n );
  test02 ( n , num_try);
  //test03 ( n );

  cout << "\n";
  n = 1000;
  test01 ( n );
  test02 ( n, num_try );
  test03 ( n );
//
//  Terminate.
//
  cout << "\n";
  cout << "SGEFA_OPENMP\n";
  cout << "  Normal end of execution.\n";

  cout << "\n";
  timestamp ( );

  return 0;
}
//****************************************************************************80

void test01 ( int n )

//****************************************************************************80
//
//  Purpose:
//
//    TEST01 runs the sequential version of SGEFA.
//
//  Modified:
//
//    07 April 2008
//
//  Author:
//
//    John Burkardt
//
{
  float *a;
  float *b;
  float err;
  int i;
  int info;
  int *ipvt;
  int job;
  int lda;
  double wtime;
  float *x;
//
//  Generate the linear system A * x = b.
//
  lda = n;
  a = new float[lda * n];
  b = new float[n];
  x = new float[n];

  matgen ( lda, n, a, x, b );
//
//  Factor the linear system.
//
  ipvt = new int[n];

  wtime = omp_get_wtime ( );
  info = sgefa ( a, lda, n, ipvt );
  wtime = omp_get_wtime ( ) - wtime;

  if ( info != 0 )
  {
    cout << "\n";
    cout << "TEST01 - Fatal error!\n";
    cout << "  SGEFA reports the matrix is singular.\n";
    exit ( 1 );
  }
//
//  Solve the linear system.
//
  job = 0;
  sgesl ( a, lda, n, ipvt, b, job );

  err = 0.0;
  for ( i = 0; i < n; i++ )
  {
    err = err + fabs ( x[i] - b[i] );
  }
  cout << "  Original  Sequential   "
       << "  " << setw(8) << n
       << "  " << setw(10) << err
       << "  " << setw(10) << wtime << "\n";

  delete [] a;
  delete [] b;
  delete [] ipvt;
  delete [] x;

  return;
}
//****************************************************************************80

void test02 ( int n , int num_try)

///****************************************************************************80
//
//  Purpose:
//
//    TEST02 runs the revised version of SGEFA in parallel.
//
//  Modified:
//
//    07 April 2008
//
//  Author:
//
//    John Burkardt
//
{
  float *a;
  float *b;
  float err;
  int i;
  int info;
  int *ipvt;
  int job;
  int lda;
  double wtime;
  float *x;
//
//  Generate the linear system A * x = b.
//
  lda = n;
  a = new float[lda * n];
  b = new float[n];
  x = new float[n];

  matgen ( lda, n, a, x, b );
//
//  Factor the linear system.
//
  ipvt = new int[n];
  int thread_num=1;
  #pragma omp parallel 
  {
	  thread_num=omp_get_num_threads();
  }

  wtime = omp_get_wtime ( );
  info = msgefa ( a, lda, n, ipvt );
  wtime = omp_get_wtime ( ) - wtime;
  
    cout<<"wtime: "<<wtime<<endl;
    ofstream myfile;
	myfile.open ("experiments/stress/sgefa_res_exp2.txt",ios::app);
	myfile<<num_try<<" , "<<thread_num<<" , "<<wtime<<endl;

  if ( info != 0 )
  {
    cout << "\n";
    cout << "TEST02 - Fatal error!\n";
    cout << "  MSGEFA reports the matrix is singular.\n";
    exit ( 1 );
  }
//
//  Solve the linear system.
//
  job = 0;
  sgesl ( a, lda, n, ipvt, b, job );

  err = 0.0;
  for ( i = 0; i < n; i++ )
  {
    err = err + fabs ( x[i] - b[i] );
  }

  cout << "  Revised     Parallel   "
       << "  " << setw(8) << n
       << "  " << setw(10) << err
       << "  " << setw(10) << wtime << "\n";

  delete [] a;
  delete [] b;
  delete [] ipvt;
  delete [] x;

  return;
}
//****************************************************************************80

void test03 ( int n )

//****************************************************************************80
//
//  Purpose:
//
//    TEST03 runs the revised version of SGEFA in sequential mode.
//
//  Modified:
//
//    07 April 2008
//
//  Author:
//
//    John Burkardt
//
{
  float *a;
  float *b;
  float err;
  int i;
  int info;
  int *ipvt;
  int job;
  int lda;
  double wtime;
  float *x;
//
//  Generate the linear system A * x = b.
//
  lda = n;
  a = new float[lda * n];
  b = new float[n];
  x = new float[n];

  matgen ( lda, n, a, x, b );
//
//  Factor the linear system.
//
  ipvt = new int[n];

  wtime = omp_get_wtime ( );
  info = msgefa2 ( a, lda, n, ipvt );
  wtime = omp_get_wtime ( ) - wtime;

  if ( info != 0 )
  {
    cout << "\n";
    cout << "TEST03 - Fatal error!\n";
    cout << "  MSGEFA2 reports the matrix is singular.\n";
    exit ( 1 );
  }
//
//  Solve the linear system.
//
  job = 0;
  sgesl ( a, lda, n, ipvt, b, job );

  err = 0.0;
  for ( i = 0; i < n; i++ )
  {
    err = err + fabs ( x[i] - b[i] );
  }

  cout << "  Revised   Sequential   "
       << "  " << setw(8) << n
       << "  " << setw(10) << err
       << "  " << setw(10) << wtime << "\n";

  delete [] a;
  delete [] b;
  delete [] ipvt;
  delete [] x;

  return;
}
//****************************************************************************80

int isamax ( int n, float x[], int incx )

//****************************************************************************80 
//
//  Purpose:
//
//    ISAMAX finds the index of the vector element of maximum absolute value.
//
//  Discussion:
//
//    WARNING: This index is a 1-based index, not a 0-based index!
//
//  Modified:
//
//    07 April 2008
//
//  Author:
//
//    FORTRAN77 original version by Lawson, Hanson, Kincaid, Krogh.
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979,
//    ISBN13: 978-0-898711-72-1,
//    LC: QA214.L56.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Algorithm 539: 
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vector.
//
//    Input, float X[*], the vector to be examined.
//
//    Input, int INCX, the increment between successive entries of SX.
//
//    Output, int ISAMAX, the index of the element of maximum
//    absolute value.
//
{
  float xmax;
  int i;
  int ix;
  int value;

  value = 0;

  if ( n < 1 || incx <= 0 )
  {
    return value;
  }

  value = 1;

  if ( n == 1 )
  {
    return value;
  }

  if ( incx == 1 )
  {
    xmax = fabs ( x[0] );

    for ( i = 1; i < n; i++ )
    {
      if ( xmax < fabs ( x[i] ) )
      {
        value = i + 1;
        xmax = fabs ( x[i] );
      }
    }
  }
  else
  {
    ix = 0;
    xmax = fabs ( x[0] );
    ix = ix + incx;

    for ( i = 1; i < n; i++ )
    {
      if ( xmax < fabs ( x[ix] ) )
      {
        value = i + 1;
        xmax = fabs ( x[ix] );
      }
      ix = ix + incx;
    }
  }

  return value;
}
//****************************************************************************80

void matgen ( int lda, int n, float a[], float x[], float b[] )

//****************************************************************************80
// 
//  Purpose:
//
//    MATGEN generates a "random" matrix for testing.
//
//  Modified:
//
//    27 April 2008
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int LDA, the leading dimension of the matrix.
//
//    Input, int N, the order of the matrix, and the length of the vector.
//
//    Output, float A[LDA*N], the matrix.
//
//    Output, float X[N], the solution vector.
//
//    Output, float B[N], the right hand side vector.
//
{
  int i;
  int j;
  int seed;
  float value;

  seed = 1325;
//
//  Set the matrix A.
//
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < n; i++ )
    {
      seed = ( 3125 * seed ) % 65536;
      value = ( ( float ) seed - 32768.0 ) / 16384.0;
      a[i+j*lda] = value;
    }
  }
//
//  Set x.
//
  for ( i = 0; i < n; i++ )
  {
    x[i] = ( float ) ( i + 1 ) / ( ( float ) n );
  }
//
//  Set b = A * x.
//
  for ( i = 0; i < n; i++ ) 
  {
    b[i] = 0.0;
    for ( j = 0; j < n; j++ )
    {
      b[i] = b[i] + a[i+j*lda] * x[j];
    }
  }
  return;
}
//****************************************************************************80

void msaxpy ( int nr, int nc, float a[], int n, float x[], float y[] )

//****************************************************************************80
//
//  Purpose:
//
//    MSAXPY carries out multiple "SAXPY" operations.
//
//  Discussion:
//
//    This routine carries out the step of Gaussian elimination where multiples
//    of the pivot row are added to the rows below the pivot row.
//
//    A single call to MSAXPY replaces multiple calls to SAXPY.
//
//  Modified:
//
//    07 April 2008
//
//  Author:
//
//    C original version by Wesley Petersen
//
//  Parameters:
//
//    Input, int NR, NC, ???
//
//    Input, float A[*], ...
//
//    Input, int N, ...
//
//    Input, float X[*], ...
//
//    Output, float Y[*], ...
//

{
  int i,j;

# pragma omp parallel \
  shared ( a, nc, nr, x, y ) \
  private ( i, j ) 

# pragma omp for 

  for ( j = 0; j < nc; j++)
  {
    for ( i = 0; i < nr; i++ )
    {
      y[i+j*n] += a[j*n] * x[i];
    }
  }
  return;
}
//****************************************************************************80

void msaxpy2 ( int nr, int nc, float a[], int n, float x[], float y[] )

//****************************************************************************80
//
//  Purpose:
//
//    MSAXPY2 carries out multiple "SAXPY" operations.
//
//  Discussion:
//
//    This routine carries out the step of Gaussian elimination where multiples
//    of the pivot row are added to the rows below the pivot row.
//
//    A single call to MSAXPY replaces multiple calls to SAXPY.
//
//  Modified:
//
//    07 April 2008
//
//  Author:
//
//    C original version by Wesley Petersen
//
//  Parameters:
//
//    Input, int NR, NC, ???
//
//    Input, float A[*], ...
//
//    Input, int N, ...
//
//    Input, float X[*], ...
//
//    Output, float Y[*], ...
//
{
  int i,j;

  for ( j = 0; j < nc; j++)
  {
    for ( i = 0; i < nr; i++ )
    {
      y[i+j*n] += a[j*n] * x[i];
    }
  }
  return;
}
//****************************************************************************80

int msgefa ( float a[], int lda, int n, int ipvt[] )

//****************************************************************************80
// 
//  Purpose:
//
//    MSGEFA factors a matrix by gaussian elimination.
//
//  Discussion:
//
//    Matrix references which would, mathematically, be written A(I,J)
//    must be written here as:
//    * A[I+J*LDA], when the value is needed, or
//    * A+I+J*LDA, when the address is needed.
//
//    This variant of SGEFA uses OpenMP for improved parallel execution.
//    The step in which multiples of the pivot row are added to individual
//    rows has been replaced by a single call which updates the entire
//    matrix sub-block.
//
//  Modified:
//
//    07 March 2008
//
//  Author:
//
//    Wesley Petersen.
//
//  Reference:
//
//    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979,
//    ISBN13: 978-0-898711-72-1,
//    LC: QA214.L56.
//
//  Parameters:
//
//    Input/output, float A[LDA*N].  On input, the matrix to be factored.
//    On output, an upper triangular matrix and the multipliers which were 
//    used to obtain it.  The factorization can be written A = L * U where
//    L is a product of permutation and unit lower triangular matrices and
//    U is upper triangular.
//
//    Input, int LDA, the leading dimension of the matrix.
//
//    Input, int N, the order of the matrix.
//
//    Output, int IPVT[N], the pivot indices.
//
//    Output, int MSGEFA, indicates singularity.
//    If 0, this is the normal value, and the algorithm succeeded.
//    If K, then on the K-th elimination step, a zero pivot was encountered.
//    The matrix is numerically not invertible.
//
{
  int info;
  int k,kp1,l,nm1;
  float t;

  info = 0;
  nm1 = n - 1;
  for ( k = 0; k < nm1; k++ )
  {
    kp1 = k + 1;
    l = isamax ( n-k, a+k+k*lda, 1 ) + k - 1;
    ipvt[k] = l + 1;

    if ( a[l+k*lda] == 0.0 )
    {
      info = k + 1;
      return info;
    }

    if ( l != k )
    {
      t          = a[l+k*lda];
      a[l+k*lda] = a[k+k*lda];
      a[k+k*lda] = t;
    }
    t = -1.0 / a[k+k*lda]; 
    sscal ( n-k-1, t, a+kp1+k*lda, 1 );
//
//  Interchange the pivot row and the K-th row.
//
    if ( l != k )
    {
      sswap ( n-k-1, a+l+kp1*lda, lda, a+k+kp1*lda, lda );
    }
//
//  Add multiples of the K-th row to rows K+1 through N.
//
    msaxpy ( n-k-1, n-k-1, a+k+kp1*lda, n, a+kp1+k*lda, a+kp1+kp1*lda );
  }

  ipvt[n-1] = n;

  if ( a[n-1+(n-1)*lda] == 0.0 )
  {
    info = n;
  }

  return info;
}
//****************************************************************************80

int msgefa2 ( float a[], int lda, int n, int ipvt[] )

//****************************************************************************80
// 
//  Purpose:
//
//    MSGEFA2 factors a matrix by gaussian elimination.
//
//  Discussion:
//
//    Matrix references which would, mathematically, be written A(I,J)
//    must be written here as:
//    * A[I+J*LDA], when the value is needed, or
//    * A+I+J*LDA, when the address is needed.
//
//    This variant of SGEFA uses OpenMP for improved parallel execution.
//    The step in which multiples of the pivot row are added to individual
//    rows has been replaced by a single call which updates the entire
//    matrix sub-block.
//
//  Modified:
//
//    07 March 2008
//
//  Author:
//
//    Wesley Petersen.
//
//  Reference:
//
//    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979,
//    ISBN13: 978-0-898711-72-1,
//    LC: QA214.L56.
//
//  Parameters:
//
//    Input/output, float A[LDA*N].  On input, the matrix to be factored.
//    On output, an upper triangular matrix and the multipliers which were 
//    used to obtain it.  The factorization can be written A = L * U where
//    L is a product of permutation and unit lower triangular matrices and
//    U is upper triangular.
//
//    Input, int LDA, the leading dimension of the matrix.
//
//    Input, int N, the order of the matrix.
//
//    Output, int IPVT[N], the pivot indices.
//
//    Output, int MSGEFA, indicates singularity.
//    If 0, this is the normal value, and the algorithm succeeded.
//    If K, then on the K-th elimination step, a zero pivot was encountered.
//    The matrix is numerically not invertible.
//
{
  int info;
  int k,kp1,l,nm1;
  float t;

  info = 0;
  nm1 = n - 1;
  for ( k = 0; k < nm1; k++ )
  {
    kp1 = k + 1;
    l = isamax ( n-k, a+k+k*lda, 1 ) + k - 1;
    ipvt[k] = l + 1;

    if ( a[l+k*lda] == 0.0 )
    {
      info = k + 1;
      return info;
    }

    if ( l != k )
    {
      t          = a[l+k*lda];
      a[l+k*lda] = a[k+k*lda];
      a[k+k*lda] = t;
    }
    t = -1.0 / a[k+k*lda]; 
    sscal ( n-k-1, t, a+kp1+k*lda, 1 );
//
//  Interchange the pivot row and the K-th row.
//
    if ( l != k )
    {
      sswap ( n-k-1, a+l+kp1*lda, lda, a+k+kp1*lda, lda );
    }
//
//  Add multiples of the K-th row to rows K+1 through N.
//
    msaxpy2 ( n-k-1, n-k-1, a+k+kp1*lda, n, a+kp1+k*lda, a+kp1+kp1*lda );
  }

  ipvt[n-1] = n;

  if ( a[n-1+(n-1)*lda] == 0.0 )
  {
    info = n;
  }

  return info;
}
//****************************************************************************80

void saxpy ( int n, float a, float x[], int incx, float y[], int incy )

//****************************************************************************80
//
//  Purpose:
//
//    SAXPY computes float constant times a vector plus a vector.
//
//  Discussion:
//
//    This routine uses unrolled loops for increments equal to one.
//
//  Modified:
//
//    23 February 2006
//
//  Author:
//
//    FORTRAN77 original version by Dongarra, Moler, Bunch, Stewart.
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979,
//    ISBN13: 978-0-898711-72-1,
//    LC: QA214.L56.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539, 
//    ACM Transactions on Mathematical Software, 
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of elements in X and Y.
//
//    Input, float A, the multiplier of X.
//
//    Input, float X[*], the first vector.
//
//    Input, int INCX, the increment between successive entries of X.
//
//    Input/output, float Y[*], the second vector.
//    On output, Y[*] has been replaced by Y[*] + A * X[*].
//
//    Input, int INCY, the increment between successive entries of Y.
//
{
  int i;
  int ix;
  int iy;
  int m;

  if ( n <= 0 )
  {
    return;
  }

  if ( a == 0.0 )
  {
    return;
  }
//
//  Code for unequal increments or equal increments
//  not equal to 1.
//
  if ( incx != 1 || incy != 1 )
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( - n + 1 ) * incx;
    }

    if ( 0 <= incy )
    {
      iy = 0;
    }
    else
    {
      iy = ( - n + 1 ) * incy;
    }

    for ( i = 0; i < n; i++ )
    {
      y[iy] = y[iy] + a * x[ix];
      ix = ix + incx;
      iy = iy + incy;
    }
  }
//
//  Code for both increments equal to 1.
//
  else
  {
    m = n % 4;

    for ( i = 0; i < m; i++ )
    {
      y[i] = y[i] + a * x[i];
    }

    for ( i = m; i < n; i = i + 4 )
    {
      y[i  ] = y[i  ] + a * x[i  ];
      y[i+1] = y[i+1] + a * x[i+1];
      y[i+2] = y[i+2] + a * x[i+2];
      y[i+3] = y[i+3] + a * x[i+3];
    }
  }

  return;
}
//****************************************************************************80

float sdot ( int n, float x[], int incx, float y[], int incy )

//****************************************************************************80
//
//  Purpose:
//
//    SDOT forms the dot product of two vectors.
//
//  Discussion:
//
//    This routine uses unrolled loops for increments equal to one.
//
//  Modified:
//
//    23 February 2006
//
//  Author:
//
//    FORTRAN77 original version by Dongarra, Moler, Bunch, Stewart
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539, 
//    ACM Transactions on Mathematical Software, 
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vectors.
//
//    Input, float X[*], the first vector.
//
//    Input, int INCX, the increment between successive entries in X.
//
//    Input, float Y[*], the second vector.
//
//    Input, int INCY, the increment between successive entries in Y.
//
//    Output, float SDOT, the sum of the product of the corresponding
//    entries of X and Y.
//
{
  int i;
  int ix;
  int iy;
  int m;
  float temp;

  temp = 0.0;

  if ( n <= 0 )
  {
    return temp;
  }
//
//  Code for unequal increments or equal increments
//  not equal to 1.
//
  if ( incx != 1 || incy != 1 )
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( - n + 1 ) * incx;
    }

    if ( 0 <= incy )
    {
      iy = 0;
    }
    else
    {
      iy = ( - n + 1 ) * incy;
    }

    for ( i = 0; i < n; i++ )
    {
      temp = temp + x[ix] * y[iy];
      ix = ix + incx;
      iy = iy + incy;
    }
  }
//
//  Code for both increments equal to 1.
//
  else
  {
    m = n % 5;

    for ( i = 0; i < m; i++ )
    {
      temp = temp + x[i] * y[i];
    }

    for ( i = m; i < n; i = i + 5 )
    {
      temp = temp + x[i  ] * y[i  ] 
                  + x[i+1] * y[i+1] 
                  + x[i+2] * y[i+2] 
                  + x[i+3] * y[i+3] 
                  + x[i+4] * y[i+4];
    }
  }

  return temp;
}
//****************************************************************************80

int sgefa ( float a[], int lda, int n, int ipvt[] )

//****************************************************************************80
//
//  Purpose:
//
//    SGEFA factors a double precision matrix by gaussian elimination.
//
//  Discussion:
//
//    Matrix references which would, mathematically, be written A(I,J)
//    must be written here as:
//    * A[I+J*LDA], when the value is needed, or
//    * A+I+J*LDA, when the address is needed.
//
//  Modified:
//
//    07 March 2008
//
//  Author:
//
//    FORTRAN77 original version by Cleve Moler.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979,
//    ISBN13: 978-0-898711-72-1,
//    LC: QA214.L56.
//
//  Parameters:
//
//    Input/output, float A[LDA*N].  On input, the matrix to be factored.
//    On output, an upper triangular matrix and the multipliers which were 
//    used to obtain it.  The factorization can be written A = L * U where
//    L is a product of permutation and unit lower triangular matrices and
//    U is upper triangular.
//
//    Input, int LDA, the leading dimension of the matrix.
//
//    Input, int N, the order of the matrix.
//
//    Output, int IPVT[N], the pivot indices.
//
//    Output, int SGEFA, indicates singularity.
//    If 0, this is the normal value, and the algorithm succeeded.
//    If K, then on the K-th elimination step, a zero pivot was encountered.
//    The matrix is numerically not invertible.
//
{
  int j;
  int info;
  int k;
  int kp1;
  int l;
  int nm1;
  float t;

  info = 0;

  for ( k = 1; k <= n - 1; k++ ) 
  {
// 
//  Find l = pivot index.
//
    l = isamax ( n-k+1, &a[k-1+(k-1)*lda], 1 ) + k - 1;
    ipvt[k-1] = l;
// 
//  Zero pivot implies this column already triangularized.
//
    if ( a[l-1+(k-1)*lda] != 0.0 ) 
    {
// 
//  Interchange if necessary.
//
      if ( l != k ) 
      {
        t                = a[l-1+(k-1)*lda];
        a[l-1+(k-1)*lda] = a[k-1+(k-1)*lda];
        a[k-1+(k-1)*lda] = t; 
      }
// 
//  Compute multipliers.
//
      t = - 1.0 / a[k-1+(k-1)*lda];
      sscal ( n-k, t, &a[k+(k-1)*lda], 1 );
// 
//  Row elimination with column indexing.
//
      for ( j = k + 1; j <= n; j++ ) 
      {
        t = a[l-1+(j-1)*lda];
        if (l != k) 
        {
          a[l-1+(j-1)*lda] = a[k-1+(j-1)*lda];
          a[k-1+(j-1)*lda] = t;
        }
        saxpy ( n-k, t, &a[k+(k-1)*lda], 1, &a[k+(j-1)*lda], 1 );
      } 
    }
    else
    { 
      info = k;
    }
  } 
  ipvt[n-1] = n;

  if (a[n-1+(n-1)*lda] == 0.0 ) 
  {
    info = n - 1;
  }
  return info;
}
//****************************************************************************80

void sgesl ( float a[], int lda, int n, int ipvt[], float b[], int job )

//****************************************************************************80
//
//  Purpose:
//
//    SGESL solves a real general linear system A * X = B.
//
//  Discussion:
//
//    SGESL can solve either of the systems A * X = B or A' * X = B.
//
//    The system matrix must have been factored by SGECO or SGEFA.
//
//    A division by zero will occur if the input factor contains a
//    zero on the diagonal.  Technically this indicates singularity
//    but it is often caused by improper arguments or improper
//    setting of LDA.  It will not occur if the subroutines are
//    called correctly and if SGECO has set 0.0 < RCOND
//    or SGEFA has set INFO == 0.
//
//  Modified:
//
//    04 April 2006
//
//  Author:
//
//    FORTRAN77 original by Dongarra, Moler, Bunch and Stewart.
//    C++ translation by John Burkardt.
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, (Society for Industrial and Applied Mathematics),
//    3600 University City Science Center,
//    Philadelphia, PA, 19104-2688.
//    ISBN: 0-89871-172-X
//
//  Parameters:
//
//    Input, float A[LDA*N], the output from SGECO or SGEFA.
//
//    Input, int LDA, the leading dimension of A.
//
//    Input, int N, the order of the matrix A.
//
//    Input, int IPVT[N], the pivot vector from SGECO or SGEFA.
//
//    Input/output, float B[N].
//    On input, the right hand side vector.
//    On output, the solution vector.
//
//    Input, int JOB.
//    0, solve A * X = B;
//    nonzero, solve A' * X = B.
//
{
  int k;
  int l;
  float t;
//
//  Solve A * X = B.
//

  if ( job == 0 )
  {
    for ( k = 1; k <= n-1; k++ )
    {
      l = ipvt[k-1];
      t = b[l-1];

      if ( l != k )
      {
        b[l-1] = b[k-1];
        b[k-1] = t;
      }
      saxpy ( n-k, t, a+k+(k-1)*lda, 1, b+k, 1 );
    }

    for ( k = n; 1 <= k; k-- )
    {
      b[k-1] = b[k-1] / a[k-1+(k-1)*lda];
      t = -b[k-1];
      saxpy ( k-1, t, a+0+(k-1)*lda, 1, b, 1 );
    }
  }
//
//  Solve A' * X = B.
//
  else
  {
    for ( k = 1; k <= n; k++ )
    {
      t = sdot ( k-1, a+0+(k-1)*lda, 1, b, 1 );
      b[k-1] = ( b[k-1] - t ) / a[k-1+(k-1)*lda];
    }

    for ( k = n-1; 1 <= k; k-- )
    {
      b[k-1] = b[k-1] + sdot ( n-k, a+k+(k-1)*lda, 1, b+k, 1 );
      l = ipvt[k-1];

      if ( l != k )
      {
        t = b[l-1];
        b[l-1] = b[k-1];
        b[k-1] = t;
      }
    }
  }
  return;
}
//****************************************************************************80

void sscal ( int n, float sa, float x[], int incx )

//****************************************************************************80
//
//  Purpose:
//
//    SSCAL scales a float vector by a constant.
//
//  Modified:
//
//    23 February 2006
//
//  Author:
//
//    FORTRAN77 original version by Lawson, Hanson, Kincaid, Krogh.
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979,
//    ISBN13: 978-0-898711-72-1,
//    LC: QA214.L56.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vector.
//
//    Input, float SA, the multiplier.
//
//    Input/output, float X[*], the vector to be scaled.
//
//    Input, int INCX, the increment between successive entries of X.
//
{
  int i;
  int ix;
  int m;

  if ( n <= 0 )
  {
  }
  else if ( incx == 1 )
  {
    m = n % 5;

    for ( i = 0; i < m; i++ )
    {
      x[i] = sa * x[i];
    }

    for ( i = m; i < n; i = i + 5 )
    {
      x[i]   = sa * x[i];
      x[i+1] = sa * x[i+1];
      x[i+2] = sa * x[i+2];
      x[i+3] = sa * x[i+3];
      x[i+4] = sa * x[i+4];
    }
  }
  else
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( - n + 1 ) * incx;
    }

    for ( i = 0; i < n; i++ )
    {
      x[ix] = sa * x[ix];
      ix = ix + incx;
    }

  }

  return;
}
//****************************************************************************80

void sswap ( int n, float x[], int incx, float y[], int incy )

//****************************************************************************80
//
//  Purpose:
//
//    SSWAP interchanges two float vectors.
//
//  Modified:
//
//    23 February 2006
//
//  Author:
//
//    FORTRAN77 original version by Lawson, Hanson, Kincaid, Krogh.
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979,
//    ISBN13: 978-0-898711-72-1,
//    LC: QA214.L56.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539, 
//    ACM Transactions on Mathematical Software, 
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vectors.
//
//    Input/output, float X[*], one of the vectors to swap.
//
//    Input, int INCX, the increment between successive entries of X.
//
//    Input/output, float Y[*], one of the vectors to swap.
//
//    Input, int INCY, the increment between successive elements of Y.
//
{
  int i;
  int ix;
  int iy;
  int m;
  float temp;

  if ( n <= 0 )
  {
  }
  else if ( incx == 1 && incy == 1 )
  {
    m = n % 3;

    for ( i = 0; i < m; i++ )
    {
      temp = x[i];
      x[i] = y[i];
      y[i] = temp;
    }

    for ( i = m; i < n; i = i + 3 )
    {
      temp = x[i];
      x[i] = y[i];
      y[i] = temp;

      temp = x[i+1];
      x[i+1] = y[i+1];
      y[i+1] = temp;

      temp = x[i+2];
      x[i+2] = y[i+2];
      y[i+2] = temp;
    }
  }
  else
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( - n + 1 ) * incx;
    }

    if ( 0 <= incy )
    {
      iy = 0;
    }
    else
    {
      iy = ( - n + 1 ) * incy;
    }

    for ( i = 0; i < n; i++ )
    {
      temp = x[ix];
      x[ix] = y[iy];
      y[iy] = temp;
      ix = ix + incx;
      iy = iy + incy;
    }
  }
  return;
}

int run_circuit ( int i);
int circuit_value ( int n, int bvec[] );
void i4_to_bvec ( int i4, int n, int bvec[] );


//****************************************************************************80

int run_circuit (int id_try )

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
# define N 27

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
# pragma omp parallel
{
  thread_num = omp_get_num_threads ( );
}
  //thread_num =4;
  //cout<<"thread num:"<<thread_num;
  solution_num = 0;
  cout<<"num_procs:"<<omp_get_num_procs();
  wtime = omp_get_wtime ( );

# pragma omp parallel \
  shared ( ihi, ilo, n, thread_num ) \
  private ( bvec, i, id, ihi2, ilo2, j, solution_num_local, value ) \
  reduction ( + : solution_num )
  {
	// cout<<"Num procs in parallel:"<< omp_get_num_threads()<<endl;
    id = omp_get_thread_num ( );
    //cout<<"#"<<id;

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
  cout<<"wtime: "<<wtime<<endl;
  ofstream myfile;
	myfile.open ("experiments/alg/circuit_alone3.txt",ios::app);
	myfile<<id_try<<" , "<<thread_num<<" , "<<wtime<<endl;
	
//
//  Terminate.
//
 
  //ti ( );

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

void openmp_program(){
	
	float a =0;
	
	#pragma omp parallel
	{
		//cout<<"check:";
		cout<<"get num threads:"<<omp_get_num_threads()<<" ";
	    for(int i=0;i<100000;i++){
			a+=i/2;
		}
	}
}


int main()
{
	benchmark_param linpack_param;
	linpack_param.n = 520000;
	linpack_param.num_tries = 25;
	linpack_param.num_thr_start = 1;
	linpack_param.num_thr_end = 4;
	
	benchmark_param gaus_seidel_param;
	gaus_seidel_param.n = 500;
	gaus_seidel_param.num_tries = 10;
	gaus_seidel_param.num_thr_start = 1;
	gaus_seidel_param.num_thr_end = 4;
	
	benchmark_param dijkstra_param;
	dijkstra_param.n = 100; // not used, fixed 600
	dijkstra_param.num_tries = 30;
	dijkstra_param.num_thr_start = 1;
	dijkstra_param.num_thr_end = 4;
	
	benchmark_param multitask_param;
	multitask_param.n = 100; // not used, fixed 2* 10000
	multitask_param.num_tries = 30;
	multitask_param.num_thr_start = 1;
	multitask_param.num_thr_end = 4;
	
	for(int i = 0; i< 600; i++){
	int n_thr = omp_get_num_threads();

	//run_circuit(i);
	//run_sgefa(i);
	//linpack_benchmark(&linpack_param);
	//gauss_seidel_benchmark(&gaus_seidel_param);
	//dijkstra_benchmark(&dijkstra_param);
	//multitask_benchmark(&multitask_param);
	
	int number_thr = i%3;
	
	if(number_thr == 0){
		number_thr=1;
	}
	else if (number_thr == 1){
		number_thr=2;
	}
	else if (number_thr == 2){
		number_thr=4;
	}
	number_thr = 3;
	
	omp_set_num_threads(number_thr);
	//run_multitask(n_thr,i);
	run_circuit(i);
	/*
	if(i%3 == 0){
		omp_set_num_threads(1);
		n_thr = 1;
	}
	else if(i%3 == 1){
		omp_set_num_threads(2);
		n_thr = 2;
	}
	else if(i%3 == 2){
		omp_set_num_threads(4);
		n_thr = 4;
	}
	*/
	
	//omp_set_num_threads(n_thr);
	//run_fft(n_thr,i);
	
	#pragma omp barrier
	
	}
	return 0;
	
}
