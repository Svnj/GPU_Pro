#include "Cloth.h"
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "common.h"
#include <iostream>
#include <string>
#include <fstream>


void updateCloth(	float3* newPos, float3* oldPos, float3* impacts, float3* velocity,					
					float deltaTime, float stepsize);

ClothSim::ClothSim() : ping(0)
{
	vboPos[0] = 0;
	vboPos[1] = 0;
	unsigned int memSize = sizeof(float) * 3 * RESOLUTION_X*RESOLUTION_Y;
	
	// Initialize mesh
	float ratio = RESOLUTION_Y / (float)RESOLUTION_X;
	float* m_hPos = new float[3 * RESOLUTION_X*RESOLUTION_Y];
	int j=0;
	for (int x=0; x<RESOLUTION_X; ++x)
	{
		for (int y=0; y<RESOLUTION_Y; ++y)
		{
			m_hPos[j*3] = x/(float)RESOLUTION_X - 0.5f;
			m_hPos[j*3+1] = 1;
			m_hPos[j*3+2] = y/(float)RESOLUTION_Y * ratio - (ratio);
			++j;
		}
	}

	// allocate device memory for intermediate impacts and velocities.
	CUDA_SAFE_CALL(cudaMalloc((void**)&devPtrImpact, memSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)&devPtrVelocity, memSize));
	cudaMemset(devPtrImpact, 0, RESOLUTION_X*RESOLUTION_Y*sizeof(float3));
	cudaMemset(devPtrVelocity, 0, RESOLUTION_X*RESOLUTION_Y*sizeof(float3));
	
	// TODO: Erzeugen der VBOs fuer die Positionen und Verbindung zu CUDA herstellen.
	glGenBuffers(2,vboPos);

	// initialise current targetBuffer
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,vboPos[1-ping]);
	glBufferData(GL_PIXEL_UNPACK_BUFFER,memSize,m_hPos,GL_DYNAMIC_DRAW);

	// initialise current source buffer
	// take values of m_hPos
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,vboPos[ping]);
	glBufferData(GL_PIXEL_UNPACK_BUFFER,memSize,m_hPos,GL_DYNAMIC_DRAW);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);

	cudaGraphicsGLRegisterBuffer(&cudaPos[1-ping],vboPos[1-ping],cudaGraphicsMapFlagsNone);

	cudaGraphicsGLRegisterBuffer(&cudaPos[ping],vboPos[ping],cudaGraphicsMapFlagsNone);

	// TODO VBO vboNormal erzeugen und mit cudaNormal verknuepfen. Das VBO braucht keine initialen Daten (NULL uebergeben). ###############################################
	//glGenBuffers(1,&vboNormal);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER,vboNormal);
	//glBufferData(GL_PIXEL_UNPACK_BUFFER,memSize,NULL,GL_DYNAMIC_DRAW);

	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);

	//cudaGraphicsGLRegisterBuffer(&cudaNormal,vboNormal,cudaGraphicsMapFlagsNone);
	
	delete[] m_hPos;
}

ClothSim::~ClothSim()
{
    CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cudaPos[0]));
    CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cudaPos[1]));
	// TODO cudaNormal freigeben #############################################################################################################################################
    glDeleteBuffers(2, (const GLuint*)vboPos);
	// TODO vboNormal freigeben ##############################################################################################################################################
	CUDA_SAFE_CALL( cudaFree( devPtrImpact ) ); 
	CUDA_SAFE_CALL( cudaFree( devPtrVelocity ) ); 
}

void ClothSim::update(GLfloat deltaTime)
{
	// Lokale Variablen, in die die Pointer auf die Daten der CUDA-Ressourcen abgelegt werden koennen.
	float* oldPos = NULL;
	float* newPos = NULL;
	float* normals = NULL;
	
	// TODO: Map cudaPos (Hinweis: cudaGraphicsMapResources)	
	cudaGraphicsMapResources( 2, cudaPos, NULL );
	//cudaGraphicsMapResources( 1, &cudaPos[ping], NULL );
	//cudaGraphicsMapResources( 1, &cudaPos[1-ping], NULL );

	// TODO: Map cudaNormal ###############################################################################################################################################
	    
	// TODO: Pointer auf die Daten von cudaPos[ping] und cudaPos[1-ping] beschaffen. (Hinweis: cudaGraphicsResourceGetMappedPointer)
	unsigned int memSize = sizeof(float) * 3 * RESOLUTION_X*RESOLUTION_Y;

	cudaGraphicsResourceGetMappedPointer( (void**)&oldPos, &memSize, cudaPos[1-ping]);
	cudaGraphicsResourceGetMappedPointer( (void**)&newPos, &memSize, cudaPos[ping]);

	// TODO: Pointer auf die Daten von cudaNormal beschaffen. #############################################################################################################

	// Launch update
	float stepSize = 0.5f; // steers how quickly the iterative refinement converges	
	updateCloth((float3*)newPos, (float3*)oldPos, (float3*)devPtrImpact, (float3*)devPtrVelocity, deltaTime, stepSize);

	// TODO: Unmap cudaNormal #############################################################################################################################################
	// TODO: Unmap cudaPos (Hinweis: cudaGraphicsUnmapResources)	
	cudaGraphicsUnmapResources(2, cudaPos, NULL);
	//cudaGraphicsUnmapResources(1, &cudaPos[ping], NULL);
	//cudaGraphicsUnmapResources(1, &cudaPos[1-ping], NULL);

	// Swap ping pong roles.
	ping = 1-ping;
}

unsigned int ClothSim::getVBOPos(unsigned int p) const
{
	return vboPos[p];
}

unsigned int ClothSim::getVBONormal() const
{
	return vboNormal;
}

unsigned int ClothSim::getResolutionX() const
{
	return RESOLUTION_X;
}

unsigned int ClothSim::getResolutionY() const
{
	return RESOLUTION_Y;
}

unsigned int ClothSim::getPingStatus() const
{
	return ping;
}