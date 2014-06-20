
#version 330

in layout(location=0) vec3 in_Position;
in layout(location=1) vec2 in_TexCoord;

out vec2 out_TexCoord;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};

void main()
{
	gl_Position = Projection * View * vec4(in_Position,1);
	out_TexCoord = in_TexCoord;
}
