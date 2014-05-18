#version 420 core

// ----------------------------------------------------------------
// Uniforms (application input)
// ----------------------------------------------------------------
layout(std140) uniform GlobalMatrices {
	mat4 Projection; // Projektionsmatrix (wir brauchen keine View)
};

layout(std140) uniform Params {
	vec2 xRange;	// Minimaler und maximaler x-Wert
	vec2 yRange;	// Minimaler und maximaler y-Wert
	float stepSize; // Integrationsschrittweite
	float time;     // Derzeitige Zeit (schreitet in jedem Frame voran)
};

// Abstands-Schwellenwert, ab dem zwischen zwei aufeinander folgenden Partikeln ein neues Partikel eingefuegt werden soll.
const float refinementThreshold = 0.015;

// 3D Textur, die das Vektorfeld enthaelt
uniform sampler3D texFlow;

// ----------------------------------------------------------------
// Input and output layout of geometry shader.
// ----------------------------------------------------------------

layout(points) in;
layout(points, max_vertices = 2) out;

#define HEAD 2
#define BODY 1
#define TAIL 0

struct PARTICLE
{
	vec2 PositionA;
	uint StateA;	// 2=Head, 1=Body, 0=Tail

	vec2 PositionB;
	uint StateB;	// 2=Head, 1=Body, 0=Tail
};

in PARTICLE vs_out[1];

out vec2 gs_out_Position;
out flat uint gs_out_State;	// 2=Head, 1=Body, 0=Tail

// ----------------------------------------------------------------
// Some helpers related to the vector data.
// ----------------------------------------------------------------

// Transforms a vertex from world space to texture space = [0..1]^2
vec2 toGrid(vec2 position) {
	vec2 tx;
	tx.x = 1- (position.x - xRange.x) / (xRange.x-xRange.y);
	tx.y = 1- (position.y - yRange.x) / (yRange.x-yRange.y);
	return tx;
}

// Reads the velocity vector at a certain position and time.
vec2 sampleVelocity(vec2 position, float t) {
	return textureLod(texFlow, vec3(toGrid(position), t), 0).xy;
}

// checks whether a position is in the domain.
bool inGrid(vec2 position) {
	return position.x > xRange.x && position.x < xRange.y && position.y > yRange.x && position.y < yRange.y;
}

// checks whether a particle is in the obstacle (cylinder is placed at (0,0) with a diameter of 0.125)
bool inObstacle(vec2 position) {
	return length(position) < 0.125;
}

// advects a particle using euler integration (as long as we are in the valid time domain)
vec2 advect(vec2 position) {
	return position + sampleVelocity(position, time) * stepSize * (time < 1 ? 1 : 0);
}

// ----------------------------------------------------------------
// The geometry shader.
// ----------------------------------------------------------------

void main(void)
{
	// --------------------------------------------
	// advect particle
	// --------------------------------------------

	// Derzeitigen Zustand uebernehmen (auf Ausgabevariablen schreiben)
	gs_out_Position = vs_out[0].PositionA;
	gs_out_State = vs_out[0].StateA;

	// TODO: Wenn nicht "Head" dann advektieren.

	// TODO: Ist das Partikel im erlaubten Domain? (Weder Domain verlassen, noch im Zylinder)
	{
		// Transformieren des Partikels auf den Viewport
		gl_Position = Projection * vec4(gs_out_Position, 0,1);

		// TODO: Wenn der Nachfolger das Domain verlassen hat, wird dieses Partikel zum neuen "Tail".
		
		// Emitieren des Partikels
		EmitVertex();
		EndPrimitive();
	
		// --------------------------------------------
		// Refinement
		// --------------------------------------------
		// TODO: Falls nicht Tail, time<1 und Distance zwischen diesem und nachfolgendem Partikel groesser ist als refinementThreshold
		{
			// TODO: Neues Body-Partikel in der Mitte (zwischen aktuellem und nachfolgendem Partikel) einfuegen
		}

	}
}
