#version 330

in vec3 in_Position;
in vec3 in_Normal;
in vec2 in_TexCoord;

varying vec4 diffuse, ambient;
varying vec3 normal, lightDir;

layout (location = 0) out vec4 fPosition;
layout (location = 1) out vec3 fNormal;
layout (location = 2) out vec4 fColor;

void main()
{	
	vec3 n = normalize(normal);
	float NdotL = abs(dot(n, normalize(lightDir)));
	gl_FragColor = diffuse * NdotL + ambient;
}
