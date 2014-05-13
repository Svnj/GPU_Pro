// geometry shader for growing hair

#version 150

#define OUT_VERTS 12

layout(triangles) in;
layout(triangle_strip, max_vertices = OUT_VERTS) out;

in vec3 normal[3];

flat out vec3 originatingVertex;

const float grav = 0.005f;
const float displacementFactor = 0.0125*2;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};

void transformAndEmitVertex()
{
	gl_Position = Projection * View * gl_Position;
	EmitVertex();
}

void placeTwoPoints(vec4 position,vec3 displacementVector, float displacementFactor)
{
	gl_Position = position;
	gl_Position -= vec4(displacementVector * displacementFactor,0);
	transformAndEmitVertex();

	gl_Position = position;
	gl_Position += vec4(displacementVector * displacementFactor,0);
	transformAndEmitVertex();
}

void main(void)
{
	vec4 position = vec4(0);
	vec4 positionStep = vec4(0);
	vec3 displacementVector = vec3(0);
	vec3 growthVector = vec3(0);
	originatingVertex = gl_in[0].gl_Position.xyz;

	for(int i=0; i< gl_in.length(); i++){
		// vertex
		originatingVertex = gl_in[i].gl_Position.xyz;
				
		position= gl_in[i].gl_Position;
		growthVector = normal[i];
		//growthVector = (inverse(View)*vec4(0.0f,-1.0f,1.0f,0.0f)).xyz;
		positionStep = (vec4(growthVector*0.1,0))/(OUT_VERTS/2);		
		//displacementVector = cross(normal[i],vec3(0,0,1));
		displacementVector = (inverse(View)*vec4(1.0f,0.0f,0.0f,0.0f)).xyz;
		normalize(displacementVector);

		placeTwoPoints(position,displacementVector,displacementFactor*2);

		for(int j=1; j< OUT_VERTS/2; j++){
			position = position + positionStep;
			position.y -= j * grav;
			placeTwoPoints(position,displacementVector,displacementFactor/j);
		}
		EndPrimitive();
	}
}
