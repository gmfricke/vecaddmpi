// Based on https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/
#include "mpi.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <time.h>

extern int DATA_DISTRIBUTE;
extern int DATA_COLLECT;

// CUDA kernel. Each thread takes care of one element of c
__global__ void gpu_vecadd(float *a, float *b, float *c, int n)
{
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;
 
  // Make sure we do not go out of bounds
  if (id < n)
    c[id] = a[id] + b[id];
}

void compute_process_gpu(unsigned int vector_size ) 
{
  clock_t begin = clock();	
  int np;
  unsigned int num_bytes = vector_size * sizeof(float);
  float *input_a, *input_b, *output;
  MPI_Status status;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  int server_process = 0;

  // This process id
  int rank = -1;
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
  int compute_node_indicator = 1; // 0 for non-compute node, 1 for compute node. Use all reduce for simplicity.
  MPI_Allreduce(&compute_node_indicator, &n_compute_nodes_on_host, 1, MPI_INT, MPI_SUM, intranode_comm);

  printf("ComputeNode (%d): There are %d MPI proceses on host %s, %d are compute processes.\n", 
	 rank, n_procs_on_host, proc_name, n_compute_nodes_on_host);
  fflush(stdout);
  
  // Report some GPU info 
  int n_gpus = 0;
  cudaGetDeviceCount(&n_gpus);

  if (n_gpus < n_compute_nodes_on_host)
    {
      printf("ComputeNode (%d): There are insufficient GPUs (%d) for the %d processes on host %s.\n", rank, n_gpus, n_procs_on_host, proc_name);
      fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

  printf("ComputeNode (%d): Detected %d GPUs on host %s.\n", rank, n_gpus, proc_name);
  fflush(stdout);

  // Select GPU to use (use mod to allocate the GPUs round-robin style using the remainder)
  unsigned int gpu_id = local_rank % n_gpus;
  cudaSetDevice( gpu_id );

  printf("ComputeNode (%d): Using GPU %d.\n", rank, gpu_id);
  fflush(stdout);
  	
  // Name the input and output vectors

  // Device (GPU) input vectors
  float *gpu_input_a;
  float *gpu_input_b;
  
  //Device output vector
  float *gpu_output;

  // Allocate memory	

  // Allocate memory for each vector on host
  input_a = (float*)malloc(num_bytes);
  input_b = (float*)malloc(num_bytes);
  output = (float*)malloc(num_bytes);
 
  // Allocate memory for each vector on the GPU
  cudaMalloc(&gpu_input_a, num_bytes);
  cudaMalloc(&gpu_input_b, num_bytes);
  cudaMalloc(&gpu_output, num_bytes);

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("ComputeProcess: Started with rank %d (%g s).\n", rank, time_spent);
  fflush(stdout);


  printf("ComputeProcess (%d): Waiting for data with %d elements from data server %d.\n", rank, vector_size, server_process);
  fflush(stdout);

  begin = clock();

  /* Get the input data from data server process */
  MPI_Recv(input_a, vector_size, MPI_FLOAT, server_process, DATA_DISTRIBUTE, MPI_COMM_WORLD, &status);

  MPI_Recv(input_b, vector_size, MPI_FLOAT, server_process, DATA_DISTRIBUTE, MPI_COMM_WORLD, &status);

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("ComputeProcess (%d): Waited for %g s.\n", rank, time_spent);

  begin = clock();

  /* Compute the partial vector addition */

  // Copy host vectors to device
  
  cudaMemcpy( gpu_input_a, input_a, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( gpu_input_b, input_b, num_bytes, cudaMemcpyHostToDevice);
 
  end = clock();	
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  printf("ComputeProcess (%d): Loaded data onto GPU (%g s).\n", rank, time_spent);
  fflush(stdout);

  begin = clock();

  int block_size, grid_size;
 
  // Number of threads in each thread block
  block_size = 1024;
 
  // Number of thread blocks in grid
  grid_size = (int)ceil((float)vector_size/block_size);
 
  // Execute the kernel
  gpu_vecadd<<<grid_size, block_size>>>(gpu_input_a, gpu_input_b, gpu_output, vector_size);

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  printf("ComputeProcess (%d): GPU partial vector addition complete (%g s).\n", rank, time_spent);
  fflush(stdout);

  // Copy array back to host
  begin = clock();     
  
  cudaMemcpy( output, gpu_output, num_bytes, cudaMemcpyDeviceToHost );
  
  printf("ComputeProcess (%d): Copy data from GPU (%g s).\n", rank, time_spent);
  fflush(stdout);

  // Signal that computation is done
  
  // Check GPU calculation against CPU version for debugging
  // Add vectors using CPU
  //float *cpu_output = (float *)malloc(num_bytes);
  //for(int i = 0; i < vector_size; i++) 
  //  cpu_output[i] = input_a[i] + input_b[i];

  //float error = 0;
  //float total = 0;
  //for(int i = 0; i < vector_size; i++)
  //  {
  //    total = total + cpu_output[i];
  //    error = error + abs(cpu_output[i] - output[i]);
  //  }

  //printf("ComputeNode (%d): GPU result differs from CPU result by %f%%.\n", rank, 100.0*error/total);
  //fflush(stdout);

  //printf("ComputeNode (%d): CPU result (first 10 elements):\n", rank);
  //for(int i = 0; i < 10; i++)
  //    printf("%f ", cpu_output[i]);
  //printf("\n");
  //fflush(stdout);	 

  //printf("ComputeNode (%d): GPU result (first 10 elements):\n", rank);
  //for(int i = 0; i < 10; i++)
  //    printf("%f ", output[i]);
  //printf("\n");
  //fflush(stdout);	 

  MPI_Barrier(MPI_COMM_WORLD);

  // Send the output to the data server
  MPI_Send(output, vector_size, MPI_FLOAT, server_process, DATA_COLLECT, MPI_COMM_WORLD);

  // Clean up memory

  // Release device memory
  cudaFree(gpu_input_a);
  cudaFree(gpu_input_b);
  cudaFree(gpu_output);
 
  // Release host memory
  free(input_a);
  free(input_b);
  free(output);
  //free(cpu_output); 
}
