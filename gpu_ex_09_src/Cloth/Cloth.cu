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
	float3 impulse = make_float3(0,0,0);
	float dist = length(p);
	
	if(dist < (SPHERE_RADIUS + SKIN_WIDTH))
	{
		float3 s = normalize(p) * (SPHERE_RADIUS + SKIN_WIDTH)*2;
		impulse = s/h;
	}
	return impulse;
}

// -----------------------------------------------------------------------------------------------
// Aufsummieren der Impulse, die von den benachbarten Gitterpunkten ausgeuebt werden.
// impacts += ...
__global__ void computeImpacts(float3* oldPos, float3* impacts, float stepsize, float h)
{
	// TODO: Positionen der benachbarten Gitterpunkte und des eigenen Gitterpunktes ablesen.
	// own position:
	int index = 0;
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	int line = blockDim.y * gridDim.y;
	index = Y + X * line;
	// neighbors (4-connected)
	int indexN0 = (Y > 0)? Y-1 + X * line : -1;
	int indexN1 = (X > 0)? Y + (X-1) * line : -1;
	int indexN2 = (Y < line-1)? Y+1 + X * line : -1;
	int indexN3 = (X < blockDim.x * gridDim.x-1)? Y + (X+1) * line : -1;

	// TODO: Kollisionsbehandlung mit Kugel durchfuehren.
	//oldPos[index] = sphereCollision(oldPos[index], h);

	// TODO: Mit jedem Nachbar besteht ein Constraint. Dementsprechend fuer jeden Nachbar 
	//		 computeImpact aufrufen und die Ergebnisse aufsummieren.
	float3 impact = sphereCollision(oldPos[index], h);
	float3 zeroVec = make_float3(0,0,0);
	// neighbors also mustn't enter the sphere
	/*impact += (indexN0 == -1)? zeroVec : computeImpact(oldPos[index], sphereCollision(oldPos[indexN0], h), stepsize, h);
	impact += (indexN1 == -1)? zeroVec : computeImpact(oldPos[index], sphereCollision(oldPos[indexN1], h), stepsize, h);
	impact += (indexN2 == -1)? zeroVec : computeImpact(oldPos[index], sphereCollision(oldPos[indexN2], h), stepsize, h);
	impact += (indexN3 == -1)? zeroVec : computeImpact(oldPos[index], sphereCollision(oldPos[indexN3], h), stepsize, h);
	*/
	if(indexN0 != -1)
		impact += computeImpact(oldPos[index], oldPos[indexN0],  stepsize, h);
	if(indexN1 != -1)
		impact += computeImpact(oldPos[index], oldPos[indexN1],  stepsize, h);
	if(indexN2 != -1)
		impact += computeImpact(oldPos[index], oldPos[indexN2],  stepsize, h);
	if(indexN3 != -1)
		impact += computeImpact(oldPos[index], oldPos[indexN3],  stepsize, h);
	// TODO: Die Summe der Impulse auf "impacts" des eigenen Gitterpunkts addieren.	
	impacts[index] += impact;
}

// -----------------------------------------------------------------------------------------------
// Preview-Step
// newpos = oldpos + (velocity + impacts * h) * h
__global__ void previewSteps(	float3* newPos, float3* oldPos, float3* impacts, float3* velocity,								
								float h)
{
	// TODO: Berechnen, wo wir waeren, wenn wir eine Integration von der bisherigen Position 
	//		 mit der bisherigen Geschwindigkeit und den neuen Impulsen durchfuehren.
	int index = 0;
    // Index berechnen	
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	index = Y + X * blockDim.y * gridDim.y;

	// TODO: don't check this here but find a proper way to not call it at all? ###########################################################################################
	if( Y == blockDim.y*gridDim.y-1)
		return;

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
	index = Y + X * blockDim.y * gridDim.y;

	// TODO: Update velocity.
	velocity[index] = velocity[index] * LINEAR_DAMPING + (impacts[index] - make_float3(0,GRAVITY,0)) * h;
}

// -----------------------------------------------------------------------------------------------
// Test-Funktion die nur dazu da ist, damit man etwas sieht, sobald die VBOs gemapped werden...
__global__ void test( float3* newPos, float3* oldPos, float h)
{
	int index = 0;
    // Index berechnen	
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	index = Y + X * blockDim.y * gridDim.y;

	// TODO: don't check this here but find a proper way to not call it at all? ###########################################################################################
	if( Y == blockDim.y*gridDim.y-1)
		return;

	newPos[index] = oldPos[index] + make_float3(0, -h/20, 0);
	newPos[index] = sphereCollision(newPos[index], h);
}

// -----------------------------------------------------------------------------------------------
void updateCloth(	float3* newPos, float3* oldPos, float3* impacts, float3* velocity,					
					float h, float stepsize)
{
	dim3 blocks(RESOLUTION_X, RESOLUTION_Y, 1);

	// -----------------------------
	// Clear impacts
	cudaMemset(impacts, 0, RESOLUTION_X*RESOLUTION_Y*sizeof(float3));

	// Updating constraints is an iterative process.
	// The more iterations we apply, the stiffer the cloth become.
	for (int i=0; i<10; ++i)
	{
		// -----------------------------
		// TODO: previewSteps Kernel aufrufen (Vorhersagen, wo die Gitterpunkte mit den aktuellen Impulsen landen wuerden.)
		// newpos = oldpos + (velocity + impacts * h) * h
		previewSteps<<<blocks,1>>>(newPos, oldPos, impacts, velocity, h);
		
		// -----------------------------
		// TODO: computeImpacts Kernel aufrufen (Die Impulse neu berechnen, sodass die Constraints besser eingehalten werden.)
		// impacts += ...
		computeImpacts<<<blocks,1>>>(newPos, impacts, stepsize, h);
	}

	// call test fuction to see if the mapping has worked
	//test<<<blocks, 1>>>(newPos, oldPos, h);

	// -----------------------------
	// TODO: Approximieren der Normalen ###################################################################################################################################
	
	// -----------------------------
	// TODO: Integrate velocity kernel ausfuehren
	// Der kernel berechnet:  velocity = velocity * LINEAR_DAMPING + (impacts - (0,GRAVITY,0)) * h 	
	integrateVelocity<<<blocks,1>>>(impacts, velocity, h);
}
