
#version 330

in vec2 out_TexCoord;
in vec3 out_Normal_cam_space;

out vec4 FragColor;

void main()
{	
	FragColor = vec4(vec3(out_Normal_cam_space.x),1);
}
