
#version 330

in vec2 out_TexCoord;

out vec4 FragColor;

void main()
{	
	FragColor = vec4(out_TexCoord * max(length(dFdx(out_TexCoord)), length(dFdy(out_TexCoord))) * 70, 0.5,1);
}
