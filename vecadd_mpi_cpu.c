//http://www.cs.ucr.edu/~nael/217-f15/lectures/217-lec21.pdf
// The code above is broken in several ways. Fixed here. 
#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

int DATA_DISTRIBUTE = 0;
int DATA_COLLECT = 1;

// Prints to the provided buffer a nice number of bytes (KB, MB, GB, etc)
int fastFloor(double x) 
{
  int xi = (int)x;
  return x < xi ? xi - 1 : xi;
}

void pretty_bytes(char* buf, uint bytes)
{
  const char* suffixes[7];
  suffixes[0] = "B";
  suffixes[1] = "KB";
  suffixes[2] = "MB";
  suffixes[3] = "GB";
  suffixes[4] = "TB";
  suffixes[5] = "PB";
  suffixes[6] = "EB";
  uint s = 0; // which suffix to use
  double count = bytes;
  while (count >= 1024 && s < 7)
    {
      s++;
      count /= 1024;
    }
  if (count - fastFloor(count) == 0.0)
    sprintf(buf, "%d %s", (int)count, suffixes[s]);
  else
    sprintf(buf, "%.1f %s", count, suffixes[s]);
}

void random_data( float *buffer, unsigned int size, float min, float max )
{
  int i;
  for( i = 0; i < size; i=i+1)
      buffer[i] = min+((float)rand()/RAND_MAX)*(max-min);
}

void data_server(unsigned int vector_size) 
{
  int np = -1;
  int rank = -1;

  /* Set MPI Communication Size */
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  // This process id
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  unsigned int num_bytes = vector_size * sizeof(float);
  char data_size[18] = "";
  pretty_bytes( data_size, num_bytes );
  printf("DataServer: Starting with rank %d and vector of size %s\n", rank, data_size);
  fflush(stdout);

  int num_nodes = np - 1, first_node = 0;
  float *input_a = 0, *input_b = 0, *output = 0;

  /* Allocate input data */
  input_a = (float *)malloc(num_bytes);
  input_b = (float *)malloc(num_bytes);
  output = (float *)malloc(num_bytes);

  if(input_a == NULL || input_b == NULL || output == NULL) 
    {
      printf("DataServer (%d) couldn't allocate memory\n", rank);
      MPI_Abort( MPI_COMM_WORLD, 1 );
    }

  /* Initialize input data */
  float min = 1, max = 10;
  printf("DataServer (%d): filling two input vectors with random floats between %f and %f...\n", rank, min, max);
  fflush(stdout);
  random_data(input_a, vector_size , min, max);
  random_data(input_b, vector_size , min, max);

  printf("DataServer (%d): finished generating random vector elements.\n", rank);
  fflush(stdout);

  /* Send data to compute nodes */
  float *ptr_a = input_a;
  float *ptr_b = input_b;
  
  int compute_portion_size; // Number of element to receive (floats)
  compute_portion_size = vector_size/num_nodes; // This is like the stride

  for(int process = 0; process < num_nodes; process++) 
    {
      printf("DataServer (%d): Sending vector of size %d to compute node with id %d.\n", rank, compute_portion_size, process);
      fflush(stdout);

      MPI_Send(ptr_a, compute_portion_size, MPI_FLOAT, process, DATA_DISTRIBUTE, MPI_COMM_WORLD);
      ptr_a += compute_portion_size;

      MPI_Send(ptr_b, compute_portion_size, MPI_FLOAT, process, DATA_DISTRIBUTE, MPI_COMM_WORLD);
      ptr_b += compute_portion_size;
    }
  
  /* Wait for nodes to compute */
  MPI_Barrier(MPI_COMM_WORLD);
  printf("DataServer (%d): All compute nodes finished. Receiving partial results.\n", rank);
  fflush(stdout);

  /* Collect output data */
  MPI_Status status;
  float *start_addr; // Where to start writing data to memory

  // Iterate over all the workers and receive their results
  for(int process = 0; process < num_nodes; process++) 
    {
      // Divide up the result buffer so each worker writes to the 
      // correct area of memory
      start_addr = output + process*compute_portion_size; // Offset by the index of this process times the stride
      
      MPI_Recv( start_addr, compute_portion_size, MPI_FLOAT, process, DATA_COLLECT, MPI_COMM_WORLD, &status );
    }


  // Check the result against a serial computation
  printf("DataServer (%d): Comparing parallel computation to serial computation...\n", rank);
  fflush(stdout);

  printf("DataServer (%d): Performing serial computation...\n", rank);
  fflush(stdout);

  // Add vectors in serial
  float *serial_output = (float *)malloc(num_bytes);
  for(int i = 0; i < vector_size; ++i) 
    serial_output[i] = input_a[i] + input_b[i];
  
  printf("DataServer (%d): Comparing results...\n", rank);
  fflush(stdout);

  float error = 0;
  float total = 0;
  for(int i = 0; i < vector_size; ++i)
    {
      total = total + serial_output[i];
      error = error + abs(serial_output[i] - output[i]);
    }

  printf("DataServer (%d): Error is %f%%.\n", rank, error/total);
  fflush(stdout);

  /* Release resources */
  free(input_a);
  free(input_b);
  free(serial_output);
  free(output);
}

void compute_node(unsigned int vector_size ) 
{
  // This process id
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("ComputeNode: Starting with rank %d.\n", rank);
  fflush(stdout);

  int np;
  unsigned int num_bytes = vector_size * sizeof(float);
  float *input_a, *input_b, *output;
  MPI_Status status;

  MPI_Comm_size(MPI_COMM_WORLD, &np);
  int server_process = np - 1;

  /* Alloc host memory */
  input_a = (float *)malloc(num_bytes);
  input_b = (float *)malloc(num_bytes);
  output = (float *)malloc(num_bytes);

  printf("ComputeNode (%d): Waiting for vectors from dataserver with rank %d...\n", rank, server_process);
  fflush(stdout);

  /* Get the input data from server process */
  MPI_Recv(input_a, vector_size, MPI_FLOAT, server_process,
	   DATA_DISTRIBUTE, MPI_COMM_WORLD, &status);

  MPI_Recv(input_b, vector_size, MPI_FLOAT, server_process,
	   DATA_DISTRIBUTE, MPI_COMM_WORLD, &status);

  /* Compute the partial vector addition */
  for(int i = 0; i < vector_size; ++i) 
      output[i] = input_a[i] + input_b[i];
  

  // Signal that computation is done
  printf("ComputeNode (%d): Partial vector addition complete.\n", rank);
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Send the output */
  MPI_Send(output, vector_size, MPI_FLOAT,
	   server_process, DATA_COLLECT, MPI_COMM_WORLD);

  /* Release memory */
  free(input_a);
  free(input_b);
  free(output);
}


int main(int argc, char *argv[]) 
{
  int vector_size = 1024 * 1024 * 100;

  int pid=-1, np=-1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  unsigned int num_bytes = vector_size * sizeof(float);
  char data_size[18] = "";
  pretty_bytes( data_size, num_bytes );
  
  if(0 == pid) printf("Will try to allocate a vector of size %s.\n", data_size);
  fflush(stdout);

  if(np < 3) 
    {
      if(0 == pid) printf("Need at least 3 processes. Only %d provided.\n", np);
      MPI_Abort( MPI_COMM_WORLD, 1 ); return 1;
    }

  if(pid < np - 1){
    printf("Assigning compute node to rank %d.\n", pid);
    fflush(stdout);
    compute_node(vector_size / (np - 1));
  }
  else
    {
      printf("Assigning data server node to rank %d.\n", pid);
      fflush(stdout);
      data_server(vector_size);
    }

  MPI_Finalize();
  return 0;
}
