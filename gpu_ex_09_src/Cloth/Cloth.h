#ifndef __CLOTH_SIMULATION__
#define __CLOTH_SIMULATION__

struct cudaGraphicsResource;

// Constants
#define RESOLUTION_X 64			// Resolution along fixed line.
#define RESOLUTION_Y 96			// Resolution perpendicular to fixed line
#define SPHERE_RADIUS 0.15f		// Radius of the sphere
#define SKIN_WIDTH 0.005f		// Skin width of the sphere (important numerical parameter!)
#define LINEAR_DAMPING 0.998f	// Linear damping of velocity
#define GRAVITY 0.24f			// Gravity applied to y-axis

// !! Changing a constant requires a rebuild of the entire solution, since .cu files 
//    don't realize that they're out of date, if a dependency is rebuild.

class ClothSim
{
	private:

		// Vertex buffer objects used to share memory among cuda and gl (stores the positions of the grid points).
		unsigned int vboPos[2];
		// same resources as in "vboPos", but accessable by cuda.
		cudaGraphicsResource* cudaPos[2];

		// Vertex buffer object containing the normals.
		unsigned int vboNormal;
		// same resource as in "vboNormal", but accessable by cuda.
		cudaGraphicsResource* cudaNormal;

		// current state of ping-pong integration (0 or 1)
		unsigned int ping;		

		// intermediate impacts in device memory
		float* devPtrImpact;
		// velocities of the grid points in device memory
		float* devPtrVelocity;

	public:

		// Constructor
		ClothSim();
		// Destructor
		~ClothSim();

		// Updates the system.
		void update(float deltaTime);

		// Gets the currently active vertex buffer object.
		unsigned int getVBOPos(unsigned int ping) const;
		// Gets the vertex buffer object containing the normals.
		unsigned int getVBONormal() const;
		// Gets the resolution in x direction.
		unsigned int getResolutionX() const;
		// Gets the resolution in y direction.
		unsigned int getResolutionY() const;
		// Gets the ping status (used to identify which vao to use during rendering)
		unsigned int getPingStatus() const;

};

#endif