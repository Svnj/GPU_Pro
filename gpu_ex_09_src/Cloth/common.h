#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL( err )( HandleError(err, __FILE__, __LINE__ ) )
#endif