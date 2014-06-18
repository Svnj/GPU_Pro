

#include "common.h"
#include <stdlib.h>
#include <GL/freeglut.h>

#define DIM 512
#define blockSize 8
#define blurRadius 6
#define effectiveBlockSize (blockSize+2*blurRadius)

float sourceColors[DIM*DIM];

texture<float,2> blurDevTex;

float *sourceDevPtr;
float *transDevPtr;
float *blurDevPtr;

float readBackPixels[DIM*DIM];

int timer = 0;

int mode = 0;

void keyboard(unsigned char key, int x, int y)
{
	if(key == '1')
		mode = 0;
	else if(key == '2')
		mode = 1;
	else if(key == '3')
		mode = 2;
	else if(key == '4')
		mode = 3;
}

__global__ void animateKernel( float *sourcePtr, float *targetPtr, int time) 
{
	int index = 0;
    // TODO: Index berechnen	
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	index = X + Y * blockDim.x * gridDim.x;

	int transX = X;
	transX += time%DIM;
	if(transX >= DIM) transX -= DIM;

	int transIndex = transX + Y * blockDim.x * gridDim.x;

	targetPtr[index] = sourcePtr[transIndex];    // simple copy
}

__global__ void blurKernelGlobal( float *sourcePtr, float *targetPtr) 
{
	// filterwidth = 51 - time 109ms
	int index = 0;
	int filterWidth = blurRadius*2+1;
    // TODO: Index berechnen	
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	index = X + Y * blockDim.x * gridDim.x;

	float value = 0.0f;

	int upperLeftFilterPosX = X - blurRadius/2;
	int upperLeftFilterPosY = Y - blurRadius/2;

	for(int i = upperLeftFilterPosX; i<upperLeftFilterPosX+filterWidth; ++i) 
	{
		for(int j = upperLeftFilterPosY; j<upperLeftFilterPosY+filterWidth; ++j) 
		{
			if( i < DIM && j < DIM && i >= 0 && j >= 0)
			{
				int sampleIndex = i + j * blockDim.x * gridDim.x;
				value += sourcePtr[sampleIndex];
			}
		}
	}

	value /= filterWidth*filterWidth;

	targetPtr[index] = value;
}

__global__ void blurKernelTexture(float *targetPtr) 
{
	// filterwidth = 51 - time 98ms
	int index = 0;
	int filterWidth = blurRadius*2+1;
    // TODO: Index berechnen	
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	index = X + Y * blockDim.x * gridDim.x;

	float value = 0.0f;

	int upperLeftFilterPosX = X - blurRadius;
	int upperLeftFilterPosY = Y - blurRadius;

	for(int i = upperLeftFilterPosY; i<upperLeftFilterPosY+filterWidth; ++i) 
	{
		for(int j = upperLeftFilterPosX; j<upperLeftFilterPosX+filterWidth; ++j) 
		{
			if( i < DIM && j < DIM && i >= 0 && j >= 0)
			{
				//int sampleIndex = i + j * blockDim.x * gridDim.x;
				value += tex2D(blurDevTex,j,i);				
			}
		}
	}

	value /= filterWidth*filterWidth;

	targetPtr[index] = value;
}

__global__ void blurKernelShared(float *sourcePtr, float *targetPtr) 
{
	// Der Ansatz BlockSize = KernelSize ist _viel_ zu langsam/gibt Crash bei halbwegs großem Kernel
	// (Lesen und Schreiben funktioniert im Prinzip aber, Rest verwerfen...)
	// !!! TODO: Was finden, das funktioniert... ######################################################################################################################
	// ggf. hilft das: http://developer.download.nvidia.com/assets/cuda/files/convolutionSeparable.pdf

	// filterwidth = 51 - time ??ms
	int filterWidth = blurRadius*2+1;

	int gridDimX = gridDim.x;
	int blockDimX = blockDim.x;
	int blockDimY = blockDim.y;
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	
    // TODO: Index berechnen
	// index of shared memory to write into
	int localIndex = threadX + threadY * blockDimX;

	// index of pixel to read from (-1 if invalid)
	int index = -1;
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	index = X + Y * blockDim.x * gridDim.x;
	
	//int X = 0;
	//int Y = 0;

	// thread may only write if it is at the center of the block
	bool mayWrite = (localIndex == blockDimX*blockDimY/2);

	// allocate shared memory
	__shared__ float cache[400];

	// write into shared memory if there is a value to read from (one write per thread)
	if( index >= 0 )
		cache[localIndex] = sourcePtr[index];
	else
		cache[localIndex] = 0;

	// make sure all threads finished writing
	__syncthreads();

	// filter (only some threads are supposed to do this)
	if( mayWrite )
	{
		float value = 0.0f;

		for(int i = 0; i<blockDimY; ++i) 
		for(int j = 0; j<blockDimX; ++j) 
		{
			int sampleIndex = i + j * blockDimX;
			value += cache[sampleIndex];
		}

		value /= filterWidth*filterWidth;
		
		// index of pixel to write into
		int destIndex =  blockX + blockY * gridDimX;

		// write result
		targetPtr[destIndex] = value;
	}
}

__global__ void blurKernelSharedTobi(float *sourcePtr, float *targetPtr) 
{
	// calculate the position in source Image
	// therefore use blockSize not BlockDim.x
	int positionInImageX = blockIdx.x * blockSize + threadIdx.x - blurRadius;
	int positionInImageY = blockIdx.y * blockSize + threadIdx.y - blurRadius;

	__shared__ float cache[effectiveBlockSize * effectiveBlockSize];

	// fill the with values from gloaal memory
	cache[threadIdx.x + threadIdx.y * effectiveBlockSize] = sourcePtr[positionInImageX + positionInImageY * DIM];

	// synchronise all threads
	__syncthreads();

	// let all kernels run which have enough neighbors for mean calculation
	int kernelSizeRightSide = effectiveBlockSize - blurRadius;
	if(threadIdx.x >= blurRadius && threadIdx.x < kernelSizeRightSide && threadIdx.y >= blurRadius && threadIdx.y < kernelSizeRightSide) 
	{
			float value = 0;
			for(int i = -blurRadius; i <= blurRadius; i++)
			{
				for(int j = -blurRadius; j <= blurRadius; j++)
				{
					value += cache[(threadIdx.x + j) + (threadIdx.y + i) * effectiveBlockSize];
				}
			}
			int filterWidth = blurRadius*2+1;
			value /= filterWidth*filterWidth;
			targetPtr[positionInImageX + positionInImageY * DIM] = value;
	}
}

void display(void)	
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// TODO: Transformationskernel auf sourceDevPtr anwenden
	dim3 grid(DIM/blockSize, DIM/blockSize);
	dim3 block(blockSize, blockSize);

	timer += 1;

	animateKernel<<<grid,block>>>(sourceDevPtr, transDevPtr, timer);

	// TODO: Zeitmessung starten (see cudaEventCreate, cudaEventRecord)
		cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	// TODO: Kernel mit Blur-Filter ausführen.
	int kernelSize = blurRadius*2+1;
	// !!!!!! TODO: Richtige Kernelgroesse fuer Shared-Mem-Variante waehlen ##################################################################################################
	dim3 sharedGrid(DIM/blockSize, DIM/blockSize);
	dim3 sharedBlock(effectiveBlockSize, effectiveBlockSize);

	switch(mode)
	{
		case 0: animateKernel<<<grid,block>>>(transDevPtr, blurDevPtr, timer); break;
		case 1: blurKernelGlobal<<<grid,block>>>(transDevPtr, blurDevPtr); break;
		case 2: blurKernelTexture<<<grid,block>>>(blurDevPtr); break;
		case 3: blurKernelSharedTobi<<<sharedGrid,sharedBlock>>>(transDevPtr, blurDevPtr); break;
	}
	
	// TODO: Zeitmessung stoppen und fps ausgeben (see cudaEventSynchronize, cudaEventElapsedTime, cudaEventDestroy)
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %3.1f ms\r", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	

	// Ergebnis zur CPU zuruecklesen
    CUDA_SAFE_CALL( cudaMemcpy( readBackPixels,
                              blurDevPtr,
                              DIM*DIM*4,
                              cudaMemcpyDeviceToHost ) );

	// Ergebnis zeichnen (ja, jetzt gehts direkt wieder zur GPU zurueck...) 
	glDrawPixels( DIM, DIM, GL_LUMINANCE, GL_FLOAT, readBackPixels );
	glutSwapBuffers();
}

// clean up memory allocated on the GPU
void cleanup() {
    CUDA_SAFE_CALL( cudaFree( sourceDevPtr ) );     
	// TODO: Aufräumen zusätzlich angelegter Ressourcen.
	CUDA_SAFE_CALL( cudaUnbindTexture(blurDevTex));
	CUDA_SAFE_CALL( cudaFree( transDevPtr ) );
	CUDA_SAFE_CALL( cudaFree( blurDevPtr ) );
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("Memory Types");
	glutKeyboardFunc(keyboard);
	glutIdleFunc(display);
	glutDisplayFunc(display);

	// mit Schachbrettmuster f�llen
	for (int i = 0 ; i < DIM*DIM ; i++) {

		int x = (i % DIM) / (DIM/8);
		int y = (i / DIM) / (DIM/8);

		if ((x + y) % 2)
			sourceColors[i] = 1.0f;
		else
			sourceColors[i] = 0.0f;
	}

	// alloc memory on the GPU
	CUDA_SAFE_CALL( cudaMalloc( (void**)&sourceDevPtr, DIM*DIM*4 ) );
    CUDA_SAFE_CALL( cudaMemcpy( sourceDevPtr, sourceColors, DIM*DIM*4, cudaMemcpyHostToDevice ) );

	// TODO: Weiteren Speicher auf der GPU f�r das Bild nach der Transformation und nach dem Blur allokieren.
	CUDA_SAFE_CALL( cudaMalloc( (void**)&transDevPtr, DIM*DIM*4 ) );

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&blurDevPtr, DIM*DIM*4 ) );

	// TODO: Binding des Speichers des Bildes an eine Textur mittels cudaBindTexture.
	CUDA_SAFE_CALL( cudaBindTexture2D(NULL,blurDevTex,transDevPtr,desc,DIM,DIM,DIM*4));

	glutMainLoop();

	cleanup();
}
