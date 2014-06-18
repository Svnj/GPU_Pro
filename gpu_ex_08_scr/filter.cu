
#include "common.h"
#include <stdlib.h>
#include <GL/freeglut.h>

#define DIM 512
#define blockSize 8
#define blurRadius 6
#define effectiveBlockSize (blockSize+2*blurRadius)

float sourceColors[DIM*DIM];
float *sourceDevPtr;
float *transDevPtr;

float readBackPixels[DIM*DIM];

int timer = 0;

void keyboard(unsigned char key, int x, int y)
{
	
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

	// TODO: Kernel mit Blur-Filter ausführen.

	// TODO: Zeitmessung stoppen und fps ausgeben (see cudaEventSynchronize, cudaEventElapsedTime, cudaEventDestroy)

	// Ergebnis zur CPU zuruecklesen
    CUDA_SAFE_CALL( cudaMemcpy( readBackPixels,
                              transDevPtr,
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
	CUDA_SAFE_CALL( cudaFree( transDevPtr ) );  
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

	// mit Schachbrettmuster füllen
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

	// TODO: Weiteren Speicher auf der GPU für das Bild nach der Transformation und nach dem Blur allokieren.
	CUDA_SAFE_CALL( cudaMalloc( (void**)&transDevPtr, DIM*DIM*4 ) );

	// TODO: Binding des Speichers des Bildes an eine Textur mittels cudaBindTexture.

	glutMainLoop();

	cleanup();
}
