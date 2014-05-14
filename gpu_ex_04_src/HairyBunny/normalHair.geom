// geometry shader for growing hair

#version 150

#define OUT_VERTS 10

layout(triangles) in;
layout(line_strip, max_vertices = OUT_VERTS) out;

in vec3 normal[3];

float grav = 0.001f;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};

void main(void)
{
	gl_Position = vec4(0);
	for(int i=0; i< gl_in.length(); i++){
		// vertex
		vec4 Position = gl_in[i].gl_Position;
		gl_Position = Projection * View * Position;
		EmitVertex();
		for(int j=1; j< OUT_VERTS; j++){
			Position = Position + (vec4(normal[i]*0.125,0)/OUT_VERTS);
			Position.y -= j * grav;
			gl_Position = Projection * View * Position;
			EmitVertex();
		}
		EndPrimitive();
	}
}
