#version 330

layout (location = 0) in vec3 in_Position;
layout (location = 1) in vec3 in_Normal;
layout (location = 2) in vec2 in_TexCoord;

out vec4 position;
out vec3 normal;
out vec4 color;

void main()
{	
	normal = in_Normal;
	gl_Position = vec4(in_Position,1);
}
