// simple fragment shader that outputs transparent white (as hair color)

#version 150

flat in vec3 originatingVertex;

out vec4 fragColor;

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{		
	fragColor.r = rand(originatingVertex.xy);
	fragColor.g = rand(originatingVertex.xz);
	fragColor.b = rand(originatingVertex.yz);
	fragColor.a = 1.0f;
}
