
#version 400

// input from input assembler.
in layout(location = 0) vec3 vs_in_Position;
in layout(location = 1) vec3 vs_in_Normal;
in layout(location = 2) vec2 vs_in_TexCoord;

// output of the vertex shader.
out vec3 vs_out_Position;
out vec3 vs_out_Normal;
out vec2 vs_out_TexCoord;

void main()
{
	// bypass all data from the input assembler.
	vs_out_Position = vs_in_Position;
	vs_out_Normal = vs_in_Normal;
	vs_out_TexCoord = vs_in_TexCoord;
	gl_Position = vec4(vs_out_Position, 1);
}
