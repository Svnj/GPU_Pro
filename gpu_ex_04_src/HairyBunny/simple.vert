#version 330

out vec4 diffuse, ambient;
out vec3 lightDir;

layout (location = 0) in vec3 in_Position;
layout (location = 1) in vec3 in_Normal;
layout (location = 2) in vec2 in_TexCoord;

out vec4 position;
out vec3 normal;
out vec4 color;

void main()
{
	vec4 vrcPos = gl_ModelViewMatrix * vec4(in_Position,1);
	vrcPos.xyz /= vrcPos.w;

	normal = gl_NormalMatrix * in_Normal;	

	lightDir = gl_LightSource[0].position.xyz - vrcPos.xyz;

	diffuse = vec4(140.0f/255.0f,80.0f/255.0f,20.0f/255.0f,1.0f) * gl_LightSource[0].diffuse; //gl_FrontMaterial.diffuse
	ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient;

	gl_Position = gl_ProjectionMatrix * vrcPos;
}
