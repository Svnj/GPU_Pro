// geometry shader for growing hair

#version 150

#define OUT_VERTS 6

layout(triangles) in;
layout(triangle_strip, max_vertices = OUT_VERTS) out;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};

void main(void)
{
	//Pass-thru!
	gl_Position = vec4(0);
	for(int i=0; i< gl_in.length(); i++){
		gl_Position = gl_in[i].gl_Position;// + vec4(1, 0, -1, 0);
		gl_Position.x += 1;
		gl_Position = Projection * View * gl_Position;
		EmitVertex();
	}
	EndPrimitive();

	gl_Position = vec4(0);
	for(int i=0; i< gl_in.length(); i++){
		//gl_Position = gl_in[i].gl_Position + vec4(-1, 0, -1, 0);
		gl_Position = Projection * View * gl_Position;
		EmitVertex();
	}
	EndPrimitive();
}
