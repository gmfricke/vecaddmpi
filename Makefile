gpu:
	nvcc -arch sm_35 -c vecadd_gpu.cu -o vecadd_gpu.o -I`mpicc --showme:incdirs`
	mpic++ -c vecadd_mpi_gpu.c -o vecadd_mpi_gpu.o
	mpic++ vecadd_mpi_gpu.o vecadd_gpu.o -lcudart -o vecadd_mpi_gpu
cpu:
	mpic++ vecadd_mpi_cpu.c -o vecadd_mpi_cpu
clean:
	rm vecadd_mpi_gpu vecadd_mpi_cpu vecadd_mpi_gpu.o vecadd_mpi_cpu.o vecadd_gpu.o  
