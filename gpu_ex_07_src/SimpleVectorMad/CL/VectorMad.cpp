
// Includes
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <sys/stat.h>
//#include <cuda.h>
#include "common.h"
#include <CL/cl.h>

// Variables
float* h_A;
float* h_B;
float* h_C;
float* h_D;
float* h_E;
cl_mem d_A = NULL;
cl_mem d_B = NULL;
cl_mem d_C = NULL;
cl_mem d_D = NULL;
cl_mem d_E = NULL;


cl_int openCLErrorID;
cl_uint numberOfDevices;
cl_platform_id platformHandle = NULL;
cl_device_id deviceHandle = NULL;
cl_context contextHandle = NULL;
cl_command_queue commandQueue = NULL;
cl_program kernelProgramm = NULL;
cl_kernel kernel = NULL;

// Functions
void Cleanup(void);
void RandomInit(float*, int);
void initCL();
char* loadKernelSourceCode(const char *, size_t *);
void printBuildLog(cl_program,cl_device_id);

// Device code
//__global__ void VecAdd(const float* A, const float* B, float* C)
//{
//    int i = threadIdx.x;
//    C[i] = A[i] + B[i];
//}

// Host code
int main(int argc, char** argv)
{
    setbuf(stdout, NULL);
    printf("Simple vector addition\n");

    initCL();

    int N = 256;
    size_t size = N * sizeof(float);    

    // Allocate input vectors h_A, h_B and h_C in host memory
    h_A = (float*)malloc(size);
    if (h_A == 0) Cleanup();
    h_B = (float*)malloc(size);
    if (h_B == 0) Cleanup();
    h_C = (float*)malloc(size);
    if (h_C == 0) Cleanup();
    h_E = (float*)malloc(size);
    if (h_E == 0) Cleanup();
    h_D = (float*)malloc(size);
    if (h_D == 0) Cleanup();
	
    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);	
    RandomInit(h_D, N);
	
    // Allocate vectors in device memory
    // Copy vectors from host memory to device memory
    d_A = clCreateBuffer(contextHandle,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size,   h_A,   &openCLErrorID);
    d_B = clCreateBuffer(contextHandle,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size,   h_B,   &openCLErrorID);
    d_C = clCreateBuffer(contextHandle,CL_MEM_READ_WRITE,                       size,   NULL,  &openCLErrorID);
    d_D = clCreateBuffer(contextHandle,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,size,   h_D,   &openCLErrorID);
    d_E = clCreateBuffer(contextHandle,CL_MEM_READ_WRITE,                       size,   NULL,  &openCLErrorID);

    size_t sourceSize;
    const char* kernelSourceCode = loadKernelSourceCode("VecAdd.cl",&sourceSize);

    kernelProgramm = clCreateProgramWithSource(contextHandle,1,&kernelSourceCode,&sourceSize,&openCLErrorID);
//    printf("clCreateProgramWithSource: %d\n",openCLErrorID);
    size_t retSourceSize;
    openCLErrorID = clGetProgramInfo(kernelProgramm,CL_PROGRAM_SOURCE,NULL,NULL,&retSourceSize);
    char retSource[retSourceSize];
    openCLErrorID = clGetProgramInfo(kernelProgramm,CL_PROGRAM_SOURCE,retSourceSize,retSource,NULL);
    printf("Kernel Source Code:\n--------------\n%s\n--------------\n",retSource);

    // compile an OpenCL Programm
    openCLErrorID = clBuildProgram(kernelProgramm,1,&deviceHandle,NULL,NULL,NULL);
    if(openCLErrorID != 0)
        printBuildLog(kernelProgramm,deviceHandle);

//    printf("clBuildProgram: %d\n",openCLErrorID);

    // create an OpenCL Kernel Object
    kernel = clCreateKernel(kernelProgramm,"VecAdd",&openCLErrorID);
//    printf("clCreateKernel: %d\n",openCLErrorID);

    openCLErrorID = clSetKernelArg(kernel,0,sizeof(cl_mem),&d_A);
    openCLErrorID = clSetKernelArg(kernel,1,sizeof(cl_mem),&d_B);
    openCLErrorID = clSetKernelArg(kernel,2,sizeof(cl_mem),&d_C);
    openCLErrorID = clSetKernelArg(kernel,3,sizeof(cl_mem),&d_D);
    openCLErrorID = clSetKernelArg(kernel,4,sizeof(cl_mem),&d_E);

    size_t globalWorkSize = 256;
    size_t localWorkSize = 1;

    // Invoke kernel
    openCLErrorID = clEnqueueNDRangeKernel(commandQueue,kernel,1,NULL,&globalWorkSize,&localWorkSize,0,NULL,NULL);
//    printf("clEnqueueNDRangeKernel: %d\n",openCLErrorID);

	// Copy result from device memory to host memory
    // h_C contains the result in host memory
    openCLErrorID = clEnqueueReadBuffer(commandQueue,d_C, CL_TRUE, 0, size, h_C, 0, NULL, NULL);
//    printf("clEnqueueReadBuffer C: %d\n",openCLErrorID);

    openCLErrorID = clEnqueueReadBuffer(commandQueue,d_E, CL_TRUE, 0, size, h_E, 0, NULL, NULL);
//    printf("clEnqueueReadBuffer E: %d\n",openCLErrorID);

    // Verify result
    // DONE: Print out E and verify the result.
    int i = 0;
    for (i = 0; i < N; ++i) 
	{
        float sum = h_A[i] + h_B[i];
        //printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
        if (fabs(h_C[i] - sum) > 1e-5)
            break;

        float sumMult = h_A[i] + h_B[i] * h_D[i];
        //printf("%f + %f * %f= %f\n", h_A[i], h_B[i],h_D[i], h_E[i]);
        if (fabs(h_E[i] - sumMult) > 1e-5)
            break;
    }
    printf("%s, i = %d\n", (i == N) ? "PASSED" : "FAILED",i);

    Cleanup();
}

void Cleanup(void)
{
    clFinish(commandQueue);

    openCLErrorID = clReleaseKernel(kernel);
    openCLErrorID = clReleaseProgram(kernelProgramm);

    // Free device memory
    openCLErrorID = clReleaseMemObject(d_E);
    openCLErrorID = clReleaseMemObject(d_D);
    openCLErrorID = clReleaseMemObject(d_C);
    openCLErrorID = clReleaseMemObject(d_B);
    openCLErrorID = clReleaseMemObject(d_A);

    openCLErrorID = clReleaseCommandQueue(commandQueue);
    openCLErrorID = clReleaseContext(contextHandle);
    openCLErrorID = clReleaseDevice(deviceHandle);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);

    // DONE: Free host memory of D and E.
    if (h_D)
        free(h_D);
    if (h_E)
        free(h_E);
        
//    printf("\nPress ENTER to exit...\n");
//    fflush( stdout);
//    fflush( stderr);
//    getchar();

    exit(0);
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void initCL()
{
    // Get a List of Available OpenCL Platforms
    cl_uint numberOfPlatforms;
    openCLErrorID = clGetPlatformIDs(1,&platformHandle,&numberOfPlatforms);
//    printf("clGetPlatformIDs: %d\n",openCLErrorID);

    // Select an OpenCL Device
    openCLErrorID = clGetDeviceIDs(platformHandle,CL_DEVICE_TYPE_GPU,1,&deviceHandle,&numberOfDevices);
//    printf("clGetDeviceIDs: %d\n",openCLErrorID);

    // create an OpenCL Context
    contextHandle = clCreateContext(NULL,1,&deviceHandle,NULL,NULL,&openCLErrorID);
//    printf("clCreateContext: %d\n",openCLErrorID);

    // create an OpenCL Command Queue
    commandQueue = clCreateCommandQueue(contextHandle,deviceHandle,0,&openCLErrorID);
//    printf("clCreateCommandQueue: %d\n",openCLErrorID);

}

char* loadKernelSourceCode(const char *filename, size_t *source_size)
{
    FILE* programHandle;
    size_t programSize;
    char *programBuffer;

    // get size of kernel source
    programHandle = fopen(filename, "r");
    if (!programHandle)
    {
        fprintf(stderr, "Failed to openCL kernel source code: %s\n", filename);
        exit(1);
    }

    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);

    // read kernel source into buffer
    programBuffer = (char*) malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);

    *source_size = programSize;
    return programBuffer;
}

void printBuildLog(cl_program kernelProgramm, cl_device_id deviceHandle)
{
    cl_build_status build_status;
    openCLErrorID = clGetProgramBuildInfo(kernelProgramm, deviceHandle, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);

    char *build_log;
    size_t ret_val_size;
    openCLErrorID = clGetProgramBuildInfo(kernelProgramm, deviceHandle, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);

    build_log = new char[ret_val_size+1];
    openCLErrorID = clGetProgramBuildInfo(kernelProgramm, deviceHandle, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    build_log[ret_val_size] = '\0';

    printf("BUILD LOG: \n %s", build_log);
}
