#version 330

varying vec4 diffuse, ambient;
varying vec3 normal, lightDir;

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

	diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
	ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient;

	gl_Position = gl_ProjectionMatrix * vrcPos;
}
