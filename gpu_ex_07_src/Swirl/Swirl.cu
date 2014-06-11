
#include <stdio.h>
#include <math.h>
#include "common.h"
#include "bmp.h"
#include <stdlib.h>
#include <GL/freeglut.h>

#define DIM 512
#define blockSize 8

size_t size;

#define PI 3.1415926535897932f
#define centerX (DIM/2)
#define centerY (DIM/2)

float sourceColors[DIM*DIM];	// host memory for source image
float readBackPixels[DIM*DIM];	// host memory for swirled image

float *sourceDevPtr;			// device memory for source image
float *swirlDevPtr;				// device memory for swirled image

__global__ void swirlKernel( float *sourcePtr, float *targetPtr ) 
{
	int index = 0;
    // TODO: Index berechnen	
	//int X = threadIdx.x + blockIdx.x * blockDim.x;
	//int Y = threadIdx.y + blockIdx.y * blockDim.y;
	//index = X + Y * blockDim.x;

	index = threadIdx.x;

	// TODO: Den swirl invertieren.

	targetPtr[index] = sourcePtr[index];    // simple copy
}

void display(void)	
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// TODO: Swirl Kernel aufrufen.
	dim3 grid(blockSize, blockSize);
	dim3 block(blockSize*blockSize, blockSize*blockSize);
	//swirlKernel<<<grid,block>>>(sourceDevPtr, swirlDevPtr);
	swirlKernel<<<1,DIM*DIM>>>(sourceDevPtr, swirlDevPtr);

	// TODO: Ergebnis zu host memory zuruecklesen.
	CUDA_SAFE_CALL( cudaMemcpy(readBackPixels, swirlDevPtr, size, cudaMemcpyDeviceToHost) );

	// Ergebnis zeichnen (ja, jetzt gehts direkt wieder zur GPU zurueck...) 
	glDrawPixels( DIM, DIM, GL_LUMINANCE, GL_FLOAT, readBackPixels );

	glutSwapBuffers();
}

// clean up memory allocated on the GPU
void cleanup() {
    CUDA_SAFE_CALL( cudaFree( sourceDevPtr ) ); 
    CUDA_SAFE_CALL( cudaFree( swirlDevPtr ) ); 
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("Simple OpenGL CUDA");
	glutIdleFunc(display);
	glutDisplayFunc(display);

	// load bitmap	
	Bitmap bmp = Bitmap("who-is-that.bmp");
	if (bmp.isValid())
	{		
		for (int i = 0 ; i < DIM*DIM ; i++) {
			sourceColors[i] = bmp.getR(i/DIM, i%DIM) / 255.0f;
		}
	}

	// TODO: allocate memory at sourceDevPtr on the GPU and copy sourceColors into it.
	size = DIM * DIM * sizeof(float);

	CUDA_SAFE_CALL( cudaMalloc((void**)&sourceDevPtr, size) );
	CUDA_SAFE_CALL( cudaMemcpy(sourceDevPtr, sourceColors, size, cudaMemcpyHostToDevice) );
	
	// TODO: allocate memory at swirlDevPtr for the unswirled image.	
	CUDA_SAFE_CALL( cudaMalloc((void**)&swirlDevPtr, size) );

	glutMainLoop();

	cleanup();
}
