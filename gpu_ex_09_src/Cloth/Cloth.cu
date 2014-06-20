
// Includes
#include "CudaMath.h"
#include "Cloth.h"

// Computes the impacts between two points that are connected by a constraint in order to satisfy the constraint a little better.
__device__ float3 computeImpact(float3 me, float3 other, float stepsize, float h)
{	
	const float aimedDistance = 1.0 / (float)RESOLUTION_X;
	float3 dir = other-me;
	float ldir = length(dir);
	if (ldir==0) return dir;
	float e = (ldir - aimedDistance) * 0.5;
	return dir/ldir * e / (h*h) * stepsize;
}

// Simple collision detection against a sphere at (0,0,0) with radius SPHERE_RADIUS and skin width SKIN_WIDTH.
__device__ float3 sphereCollision(float3 p, float h)
{
	// TODO: Testen, ob Punkt im inneren der Kugel ist. Wenn ja, dann einen Impuls berechnen, der sie wieder heraus bewegt.
	if(length(p) < SPHERE_RADIUS)
	{
		p = normalize(p)*(SPHERE_RADIUS + SKIN_WIDTH);
	}
	return p;
}

// -----------------------------------------------------------------------------------------------
// Aufsummieren der Impulse, die von den benachbarten Gitterpunkten ausgeübt werden.
// impacts += ...
__global__ void computeImpacts(float3* oldPos, float3* impacts, float stepsize, float h)
{
	// TODO: Positionen der benachbarten Gitterpunkte und des eigenen Gitterpunktes ablesen.
	
	// TODO: Kollisionsbehandlung mit Kugel durchführen.

	// TODO: Mit jedem Nachbar besteht ein Constraint. Dementsprechend für jeden Nachbar 
	//		 computeImpact aufrufen und die Ergebnisse aufsummieren.

	// TODO: Die Summe der Impulse auf "impacts" des eigenen Gitterpunkts addieren.	
}

// -----------------------------------------------------------------------------------------------
// Preview-Step
// newpos = oldpos + (velocity + impacts * h) * h
__global__ void previewSteps(	float3* newPos, float3* oldPos, float3* impacts, float3* velocity,								
								float h)
{
	// TODO: Berechnen, wo wir wären, wenn wir eine Integration von der bisherigen Position 
	//		 mit der bisherigen Geschwindigkeit und den neuen Impulsen durchführen.
	int index = 0;
    // Index berechnen	
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	index = X + Y * blockDim.x * gridDim.x;

	newPos[index] = oldPos[index] + (velocity[index] + impacts[index] * h) * h;
}

// -----------------------------------------------------------------------------------------------
// Integrate velocity
// velocity = velocity * LINEAR_DAMPING + (impacts - (0,GRAVITY,0)) * h 
__global__ void integrateVelocity(	float3* impacts, float3* velocity, float h)
{
	int index = 0;
    // Index berechnen	
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	index = X + Y * blockDim.x * gridDim.x;

	// TODO: Update velocity.
	velocity[index] = velocity[index] * LINEAR_DAMPING + (impacts[index] - (0,GRAVITY,0)) * h;
}

// -----------------------------------------------------------------------------------------------
// Test-Funktion die nur dazu da ist, damit man etwas sieht, sobald die VBOs gemapped werden...
__global__ void test( float3* newPos, float3* oldPos, float h)
{
	int index = 0;
    // Index berechnen	
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	index = X + Y * blockDim.x * gridDim.x;

	newPos[index] = oldPos[index] + make_float3(0, -h, 0);

	//newPos[blockIdx.x] = oldPos[blockIdx.x] + make_float3(0, -h, 0);
}

// -----------------------------------------------------------------------------------------------
void updateCloth(	float3* newPos, float3* oldPos, float3* impacts, float3* velocity,					
					float h, float stepsize)
{
	dim3 blocks(RESOLUTION_X, RESOLUTION_Y-1, 1);

	// -----------------------------
	// Clear impacts
	cudaMemset(impacts, 0, RESOLUTION_X*RESOLUTION_Y*sizeof(float3));

	// Updating constraints is an iterative process.
	// The more iterations we apply, the stiffer the cloth become.
	for (int i=0; i<10; ++i)
	{
		// -----------------------------
		// TODO: previewSteps Kernel aufrufen (Vorhersagen, wo die Gitterpunkte mit den aktuellen Impulsen landen würden.)
		// newpos = oldpos + (velocity + impacts * h) * h
		previewSteps<<<blocks,1>>>(newPos, oldPos, impacts, velocity, h);
		
		// -----------------------------
		// TODO: computeImpacts Kernel aufrufen (Die Impulse neu berechnen, sodass die Constraints besser eingehalten werden.)
		// impacts += ...
	}

	// call test fuction to see if the mapping has worked
	//test<<<blocks, 1>>>(newPos, oldPos, h);

	// -----------------------------
	// TODO: Approximieren der Normalen
	
	// -----------------------------
	// TODO: Integrate velocity kernel ausführen
	// Der kernel berechnet:  velocity = velocity * LINEAR_DAMPING + (impacts - (0,GRAVITY,0)) * h 	
	integrateVelocity<<<blocks,1>>>(impacts, velocity, h);
}