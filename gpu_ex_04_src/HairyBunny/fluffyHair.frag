// simple fragment shader that outputs transparent white (as hair color)

#version 150

flat in vec3 originatingVertex;

out vec4 fragColor;

float rand(vec2 seed){
    return fract(sin(dot(seed.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{	
	// here i try to get some random brownish colours
	// i use the vertex from which the hair strand originated as the seed 
	// because i dont want the hair strand to change colour when i change the view 	
	fragColor.r = (130 + (40 * rand(originatingVertex.xy)))/255;
	fragColor.g = (60 + (30 * rand(originatingVertex.xz)))/255;
	fragColor.b = (20 + (30 * rand(originatingVertex.yz)))/255;
	fragColor.a = 1.0f;
}
