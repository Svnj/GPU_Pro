
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

float a;						// user parameter for picture transformation
float b;						// user parameter for picture transformation

__global__ void swirlKernel( float *sourcePtr, float *targetPtr, float a, float b) 
{
	int index = 0;
    // TODO: Index berechnen	
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	index = X + Y * blockDim.x * gridDim.x;

	// TODO: Den swirl invertieren.
	// Adjust to centered coordinate system
	int relX = X - centerX;	// x position relativ to center
	int relY = Y - centerY;	// y position relativ to center
	float originalAngle;	//angle bevore transformation, seen from a center based coordinate system

		if (relX != 0)
		{
			originalAngle = atan(((float)abs(relY)/(float)abs(relX)));

			if ( relX > 0 && relY < 0) originalAngle = 2.0f*PI - originalAngle;

			else if (relX <= 0 && relY >=0) originalAngle = PI-originalAngle;

			else if (relX <=0 && relY <0) originalAngle += PI;
		}

		else
		{
			// Take care of rare special case
			if (relY >= 0) originalAngle = 0.5f * PI;

			else originalAngle = 1.5f * PI;
		}

	// Calculate Rotation angle
	float r = sqrt((float)(relX*relX + relY*relY));
	float alpha = a * pow(r, b);

	float transformedAngle = originalAngle + alpha;

	// Transform Pixel Coordinates
	int transX = (int)(floor(r * cos(transformedAngle) + 0.5f)) + centerX;
	int transY = (int)(floor(r * sin(transformedAngle) + 0.5f)) + centerY;
	//int transX = relX + centerX;
	//int transY = relY + centerY;

	// Clamping
	if(transX < 0) transX = 0;
	if(transX >= DIM) transX = DIM-1;
	if(transY < 0) transY = 0;
	if(transY >= DIM) transY = DIM-1;

	// new Index
	int transIndex = transX + transY * blockDim.x * gridDim.x;

	targetPtr[transIndex] = sourcePtr[index];    // simple copy
}

void display(void)	
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// TODO: Swirl Kernel aufrufen.
	dim3 grid(blockSize*blockSize, blockSize*blockSize);
	dim3 block(blockSize, blockSize);
	swirlKernel<<<grid,block>>>(sourceDevPtr, swirlDevPtr, a, b);

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

void keyboard(int key, int x, int y)
{
	switch (key)
	{
		case GLUT_KEY_LEFT: a -= 0.01;
			if(a < -2) a = -2;
			printf("parameter a: %.2f , parameter b: %.3f 	\r", a, b );
			break;
		case GLUT_KEY_RIGHT: a += 0.01;
			if(a > 2) a = 2;
			printf("parameter a: %.2f , parameter b: %.3f 	\r", a, b );
			break;
		case GLUT_KEY_DOWN: b -= 0.001;
			if(b < 0) b = 0;
			printf("parameter a: %.2f , parameter b: %.3f 	\r", a, b );
			break;
		case GLUT_KEY_UP: b += 0.001;
			if(b > 1) b = 1;
			printf("parameter a: %.2f , parameter b: %.3f 	\r", a, b );
			break;
	}

}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("Simple OpenGL CUDA");
	glutSpecialFunc(keyboard);
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

	// Initalize parameters a and b
	a = 0.3;
	b = 0.7;

	printf("use arrow keys to change parameters a and b \n" );

	// TODO: allocate memory at sourceDevPtr on the GPU and copy sourceColors into it.
	size = DIM * DIM * sizeof(float);

	CUDA_SAFE_CALL( cudaMalloc((void**)&sourceDevPtr, size) );
	CUDA_SAFE_CALL( cudaMemcpy(sourceDevPtr, sourceColors, size, cudaMemcpyHostToDevice) );
	
	// TODO: allocate memory at swirlDevPtr for the unswirled image.	
	CUDA_SAFE_CALL( cudaMalloc((void**)&swirlDevPtr, size) );

	glutMainLoop();

	cleanup();
}
