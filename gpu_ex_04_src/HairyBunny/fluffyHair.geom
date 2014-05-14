// geometry shader for growing hair

#version 150

#define OUT_VERTS 12

layout(triangles) in;
layout(triangle_strip, max_vertices = OUT_VERTS) out;

in vec3 normal[3];

flat out vec3 originatingVertex;
out vec3 geomNormalInEyeSpace;
out vec3 geomPositionInEyeSpace;

const float grav = 0.005f;
const float displacementFactor = 0.0125*4;
const float displacementFactorStep = displacementFactor/((OUT_VERTS/2)-1);

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};

void transformAndEmitVertex()
{
	vec4 geomPositionInEyeSpace4 = View * gl_Position;	
	geomPositionInEyeSpace = geomPositionInEyeSpace.xyz;
	gl_Position = Projection * geomPositionInEyeSpace4;
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
		geomNormalInEyeSpace = (View*vec4(geomNormalInEyeSpace,0.0f)).xyz;		

		placeTwoPoints(position,displacementVector,displacementFactor);
		
		vec3 oldPosition;
		for(int j=1; j< OUT_VERTS/2; j++){
			oldPosition = position.xyz;
			
			// calculate new position			
			position = position + positionStep;
			position.y -= j * grav;
			
			// calculate normals for lighting
			geomNormalInEyeSpace = cross(displacementVector,normalize(position.xyz - oldPosition));
			normalize(geomNormalInEyeSpace);
			geomNormalInEyeSpace = (View*vec4(geomNormalInEyeSpace,0.0f)).xyz;

			placeTwoPoints(position,displacementVector,displacementFactor-(displacementFactorStep*j));
		}
		EndPrimitive();
	}
}
