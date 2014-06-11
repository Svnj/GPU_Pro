#include <stdio.h>
#include <math.h>
#include "common.h"
#include "bmp.h"
#include <stdlib.h>
#include <GL/freeglut.h>
#include <CL/cl.h>

#define DIM 512
#define blockSize 8

#define PI 3.1415926535897932f
#define centerX (DIM/2)
#define centerY (DIM/2)
#define CL_DIM 2

float sourceColors[DIM*DIM];	// host memory for source image
float readBackPixels[DIM*DIM];	// host memory for swirled image

cl_mem sourceDevPtr;			// device memory for source image
cl_mem swirlDevPtr;				// device memory for swirled image

cl_int openCLErrorID;
cl_uint numberOfDevices;
cl_platform_id platformHandle = NULL;
cl_device_id deviceHandle = NULL;
cl_context contextHandle = NULL;
cl_command_queue commandQueue = NULL;
cl_program kernelProgramm = NULL;
cl_kernel kernel = NULL;

size_t globalWorkSize[3] = {512,512,1};
size_t localWorkSize[3] = {16,16,1};

float a = 0.07;
float b = 1.0f;

void initCL();
char* loadKernelSourceCode(const char *, size_t *);
void printBuildLog(cl_program,cl_device_id);
void createKernel(const char *, const char *);

void display(void)	
{
    int bla = 4;
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // DONE: Swirl Kernel aufrufen.
    openCLErrorID = clEnqueueNDRangeKernel(commandQueue,kernel,2,NULL,globalWorkSize,localWorkSize,0,NULL,NULL);
    if(openCLErrorID!=0)
        printf("clEnqueueNDRangeKernel: %d\n",openCLErrorID);

    // DONE: Ergebnis zu host memory zuruecklesen.
    openCLErrorID = clEnqueueReadBuffer(commandQueue,swirlDevPtr, CL_TRUE, 0, DIM*DIM*sizeof(float), readBackPixels, 0, NULL, NULL);
    if(openCLErrorID!=0)
        printf("clEnqueueReadBuffer: %d\n",openCLErrorID);

	// Ergebnis zeichnen (ja, jetzt gehts direkt wieder zur GPU zurueck...) 
	glDrawPixels( DIM, DIM, GL_LUMINANCE, GL_FLOAT, readBackPixels );

	glutSwapBuffers();
}

// DONE: clean up memory allocated on the GPU
void cleanup() {
    clFinish(commandQueue);

    openCLErrorID = clReleaseKernel(kernel);
    openCLErrorID = clReleaseProgram(kernelProgramm);

    // Free device memory
    openCLErrorID = clReleaseMemObject(sourceDevPtr);
    openCLErrorID = clReleaseMemObject(swirlDevPtr);

    openCLErrorID = clReleaseCommandQueue(commandQueue);
    openCLErrorID = clReleaseContext(contextHandle);
    openCLErrorID = clReleaseDevice(deviceHandle);
}

void keyboard(unsigned char key, int x, int y)
{
    float delta = 0.001;
    switch (key)
    {
        case '1':
            a = a + delta;	a = a < -2.0f ? -2.0f : a;	a = a > 2.0f ? 2.0f : a;
            break;
        case '2':
            a = a - delta;	a = a < -2.0f ? -2.0f : a;	a = a > 2.0f ? 2.0f : a;
            break;
        case '3':
            b = b + delta;	b = b < 0.0f ? 0.0f : b;	b = b > 1.0f ? 1.0f : b;
            break;
        case '4':
            b = b - delta;	b = b < 0.0f ? 0.0f : b;	b = b > 1.0f ? 1.0f : b;
            break;
    }
    printf("a: %f,b: %f\n",a,b);
    openCLErrorID = clSetKernelArg(kernel,2,sizeof(cl_float),&a);
    openCLErrorID = clSetKernelArg(kernel,3,sizeof(cl_float),&b);
    //glutPostRedisplay();
}

int main(int argc, char **argv)
{
    setbuf(stdout, NULL);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(DIM, DIM);
    glutCreateWindow("Simple OpenGL OpenCL");
	glutIdleFunc(display);
	glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);

    initCL();
    createKernel("swirl.cl","swirlKernelSCB");

    size_t addressbits,localSize,computeUnits,globalSize;
    openCLErrorID = clGetDeviceInfo(deviceHandle, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &computeUnits, NULL);
    openCLErrorID = clGetDeviceInfo(deviceHandle, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &localSize, NULL);
    openCLErrorID = clGetDeviceInfo(deviceHandle, CL_DEVICE_ADDRESS_BITS, sizeof(size_t), &addressbits, NULL);
    printf("CL_DEVICE_MAX_COMPUTE_UNITS: %lu\nCL_DEVICE_MAX_WORK_GROUP_SIZE: %lu\nCL_DEVICE_ADDRESS_BITS: %lu\n",computeUnits,localSize,addressbits);

	// load bitmap	
	Bitmap bmp = Bitmap("who-is-that.bmp");
	if (bmp.isValid())
	{		
		for (int i = 0 ; i < DIM*DIM ; i++) {
			sourceColors[i] = bmp.getR(i/DIM, i%DIM) / 255.0f;
		}
    }else{
        printf("couldnt load who-is-that.bmp");
        exit(0);
    }

    // DONE: allocate memory at sourceDevPtr on the GPU and copy sourceColors into it.
    sourceDevPtr = clCreateBuffer(  contextHandle,  CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,  DIM*DIM*sizeof(float),    sourceColors,   &openCLErrorID);
    // DONE: allocate memory at swirlDevPtr for the unswirled image.
    swirlDevPtr = clCreateBuffer(   contextHandle,  CL_MEM_READ_WRITE,                      DIM*DIM*sizeof(float),    NULL,           &openCLErrorID);

    //DONE: Set Kernel Arguments
    openCLErrorID = clSetKernelArg(kernel,0,sizeof(cl_mem),&sourceDevPtr);
    openCLErrorID = clSetKernelArg(kernel,1,sizeof(cl_mem),&swirlDevPtr);
    openCLErrorID = clSetKernelArg(kernel,2,sizeof(cl_float),&a);
    openCLErrorID = clSetKernelArg(kernel,3,sizeof(cl_float),&b);

	glutMainLoop();

	cleanup();
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

void createKernel(const char* fileName,const char* kernelName){
    size_t sourceSize;
    const char* kernelSourceCode = loadKernelSourceCode(fileName,&sourceSize);

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
    {
        printBuildLog(kernelProgramm,deviceHandle);
        exit(0);
    }

    //    printf("clBuildProgram: %d\n",openCLErrorID);

    // create an OpenCL Kernel Object
    kernel = clCreateKernel(kernelProgramm,kernelName,&openCLErrorID);
    //    printf("clCreateKernel: %d\n",openCLErrorID);
}
