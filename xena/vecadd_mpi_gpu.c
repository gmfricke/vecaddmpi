// Matthew Fricke, Jun 13th, 2022
//http://www.cs.ucr.edu/~nael/217-f15/lectures/217-lec21.pdf
// This code is based on the above valuble presentation by David Kirk/NVIDIA and Wen-meiW. Hwu, 2007-2012, 
// versions of which appear in several places in the internet. 
// The code in those presentions is broken in several ways (copy paste errors, typos, missing barriers etc). 
// I've fixed those errors and tested the resulting code. I use a straight forward cuda kernel to avoid having
// to include gmac libaries. 
// This code is intended to be a straight forward and intelligible implementation of vector addition that combines
// MPI and CUDA to be used for educational and testing purposed. It is not intended to be an efficient implementation.
// Iterating MPI receives for example is inefficient but easy to understand.

// Changed the code to allocate the data server to the 0 rank process - this is just because
// slurm will put the data server on the first node instead of orphaning it on the last node
// without any compute nodes. This ensures that compure nodes are distributed evenly - important if distributed GPUs are used.

// This program enforces that each MPI process use only one GPU (assumeing GPUs support was enabled). 
//This is done by counting the number of compute nodes running on each host and making sure there are at 
// least that number of GPUs on the host (host meaning computer in this context). Then allocate GPUs to compute nodes
// using the remainder of the dividing the number of compute nodes into the number of GPUs as the ID of the GPU to use.

#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

extern void compute_node_gpu(unsigned int vector_size ); 

int DATA_DISTRIBUTE = 0;
int DATA_COLLECT = 1;

bool gpu = true;

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
  for(int i = 0; i < size; i=i+1)
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

  // First figure out how many other MPI processes are running on this node  
  MPI_Comm intranode_comm;
  int local_rank = -1;
  
  MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &intranode_comm);
  MPI_Comm_rank(intranode_comm, &local_rank);

  int n_procs_on_host;
  MPI_Comm_size( intranode_comm, &n_procs_on_host );
  
  // Get the processor (usually network host) name
  char proc_name[MPI_MAX_PROCESSOR_NAME];
  int proc_name_len = -1;
  MPI_Get_processor_name(proc_name, &proc_name_len);

  // Perform a sum reduce to determine how many of the MPI processes on this computer are compute nodes
  int n_compute_nodes_on_host = 0;
  int compute_node_indicator = 0; // 0 for non-compute node, 1 for compute node. Use all reduce instead of reduce for simplicity.
  MPI_Allreduce(&compute_node_indicator, &n_compute_nodes_on_host, 1, MPI_INT, MPI_SUM, intranode_comm);

  printf("DataServer (%d): There are %d MPI proceses on host %s, %d are compute nodes.\n", 
	 rank, n_procs_on_host, proc_name, n_compute_nodes_on_host);
  fflush(stdout);

  unsigned int num_bytes = vector_size * sizeof(float);
  char data_size[18] = "";
  pretty_bytes( data_size, num_bytes );
  printf("DataServer: Starting with rank %d.\n", rank);
  fflush(stdout);

  int n_compute_nodes = np-1;
  float *input_a = 0, *input_b = 0, *output = 0;

  /* Allocate input data */
  input_a = (float *)malloc(num_bytes);
  input_b = (float *)malloc(num_bytes);
  output = (float *)malloc(num_bytes);

  if(input_a == NULL || input_b == NULL || output == NULL) 
    {
      printf("DataServer (%d) couldn't allocate memory.\n", rank);
      MPI_Abort( MPI_COMM_WORLD, 1 );
    }

  /* Initialize input data */
  float min = 1, max = 10;
  printf("DataServer (%d): filling two input vectors with random floats between %f and %f...\n", rank, min, max);
  fflush(stdout);
  random_data(input_a, vector_size , min, max);
  random_data(input_b, vector_size , min, max);

  printf("DataServer (%d): finished generating %d random vector elements.\n", rank, vector_size);
  fflush(stdout);

  /* Send data to compute nodes */
  float *ptr_a = input_a;
  float *ptr_b = input_b;
  
  int compute_portion_size; // Number of element to receive (floats)
  compute_portion_size = vector_size/n_compute_nodes; // This is like the stride

  for(int process = 1; process < np; process++) 
    {
      printf("DataServer (%d): Sending vector A of size %d to compute node with id %d.\n", rank, compute_portion_size, process);
      fflush(stdout);

      MPI_Send(ptr_a, compute_portion_size, MPI_FLOAT, process, DATA_DISTRIBUTE, MPI_COMM_WORLD);
      ptr_a += compute_portion_size;

      printf("DataServer (%d): Sending vector B of size %d to compute node with id %d.\n", rank, compute_portion_size, process);
      fflush(stdout);

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
  for(int process = 1; process < np; process++) 
    {
      // Divide up the result buffer so each worker writes to the 
      // correct area of memory
      start_addr = output + (process-1)*compute_portion_size; // Offset by the index of this process times the stride
      
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

  printf("DataServer (%d): Error is %f%%.\n", rank, 100.0*error/total);
  fflush(stdout);

  printf("DataServer (%d): MPI result (first 10 elements):\n", rank);
  for(int i = 0; i < 10; i++)
      printf("%f ", output[i]);
  printf("\n");

  printf("DataServer (%d): serial result (first 10 elements):\n", rank);
  for(int i = 0; i < 10; i++)
      printf("%f ", serial_output[i]);
  printf("\n");
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
  int server_process = 0;

  // First figure out how many other MPI processes are running on this node  
  MPI_Comm intranode_comm;
  int local_rank = -1;
  
  MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &intranode_comm);
  MPI_Comm_rank(intranode_comm, &local_rank);

  int n_procs_on_host;
  MPI_Comm_size( intranode_comm, &n_procs_on_host );

  // Get the processor (usually network host) name
  char proc_name[MPI_MAX_PROCESSOR_NAME];
  int proc_name_len = -1;
  MPI_Get_processor_name(proc_name, &proc_name_len);
  
  // Perform a sum reduce to determine how many of the MPI processes on this computer are compute nodes
  int n_compute_nodes_on_host = 0;
  int compute_node_indicator = 1; // 0 for non-compute node, 1 for compute node. Use all reduce instead of reduce for simplicity.
  MPI_Allreduce(&compute_node_indicator, &n_compute_nodes_on_host, 1, MPI_INT, MPI_SUM, intranode_comm);

  printf("ComputeNode (%d): There are %d MPI proceses on host %s, %d are compute nodes.\n", 
	 rank, n_procs_on_host, proc_name, n_compute_nodes_on_host);
  fflush(stdout);

  /* Alloc host memory */
  input_a = (float *)malloc(num_bytes);
  input_b = (float *)malloc(num_bytes);
  output = (float *)malloc(num_bytes);

  printf("ComputeNode (%d): Waiting for vectors with %d elements from dataserver with rank %d...\n", rank, vector_size, server_process);
  fflush(stdout);

  /* Get the input data from server process */
  MPI_Recv(input_a, vector_size, MPI_FLOAT, server_process, DATA_DISTRIBUTE, MPI_COMM_WORLD, &status);

  printf("ComputeNode (%d): Received input A from data server.\n", rank);
  fflush(stdout);

  MPI_Recv(input_b, vector_size, MPI_FLOAT, server_process,
	   DATA_DISTRIBUTE, MPI_COMM_WORLD, &status);

  printf("ComputeNode (%d): Received input B from data server.\n", rank);
  fflush(stdout);

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
  // Deal with command line arguments
  for (int i = 1; i < argc; i++)  /* Skip argv[0] (program name). */
    {
      if (strcmp(argv[i], "--gpus") == 0)  /* Process optional arguments. */
	  gpu = true;
      else if (strcmp(argv[i], "--nogpus") == 0)  /* Process optional arguments. */
	  gpu = false;
      else
	{
	  /* Process non-optional arguments here. */
	}
    }
  
  int vector_size = 1024 * 1024 * 100;
  
  int rank=-1, np=-1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  unsigned n_compute_nodes = np-1;
  unsigned int num_bytes = vector_size * sizeof(float);
  char data_size[18] = "";
  pretty_bytes( data_size, num_bytes );
  
  // Let the user know whether or not GPU support was enabled
  if(rank == 0) printf("GPU support set to:%s.\n", gpu ? "\033[0;32m true\033[0m" : "\033[0;33m false\033[0m");
  fflush(stdout);

  if(0 == rank) printf("Dataserver will try to allocate 3 vectors of size %s each.\n", data_size);
  fflush(stdout);

  if(np < 3) 
    {
      if(0 == rank) printf("Need at least 3 processes. Only %d provided.\n", np);
      MPI_Abort( MPI_COMM_WORLD, 1 ); return 1;
    }

  if(rank == 0)
    {
      printf("Assigning data server node to rank %d.\n", rank);
      fflush(stdout);
      data_server(vector_size);
    }
  else
    {
      printf("Assigning compute node to rank %d.\n", rank);
      fflush(stdout);
      if(gpu)
	compute_node_gpu(vector_size/n_compute_nodes);
      else
	compute_node(vector_size/n_compute_nodes);

    }

  MPI_Finalize();
  return 0;
}
