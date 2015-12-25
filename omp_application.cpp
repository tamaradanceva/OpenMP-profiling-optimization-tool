#include <iostream>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include "omp.h"
#include <fstream>
#include <iostream>
# include <cstdlib>
# include <iomanip>
# include <ctime>
# include <omp.h>

using namespace std;

int run_circuit ( );
int circuit_value ( int n, int bvec[] );
void i4_to_bvec ( int i4, int n, int bvec[] );
void timestamp ( );

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
  cout<<"thread num:"<<thread_num;
  solution_num = 0;

  wtime = omp_get_wtime ( );

# pragma omp parallel \
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
  cout<<"wtime: "<<wtime<<endl;
  ofstream myfile;
	myfile.open ("predict_345_circuit1.txt",ios::app);
	myfile<<id_try<<" , "<<omp_get_num_threads()<<" , "<<thread_num<<" , "<<wtime<<endl;
	
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
//****************************************************************************80

void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    24 September 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

 // cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
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
	
	for(int i = 0; i< 20; i++){
	int n_thr = omp_get_num_threads();
	omp_set_num_threads(4);
	
	run_circuit(i);
	
	
	}
	return 0;
	
}
