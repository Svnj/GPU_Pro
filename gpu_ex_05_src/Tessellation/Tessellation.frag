
#version 400

// output of the domain shader.
in vec2 ds_out_TexCoord;
in vec3 ds_out_Normal;

// output color.
out vec4 ps_out_FragColor;

void main()
{
	vec3 n = normalize(ds_out_Normal);
	vec3 l = vec3(0,0,1);

	float NdotL = abs(dot(n,l));
	NdotL = NdotL * 0.8 + pow(NdotL, 16);

	ps_out_FragColor = vec4(NdotL, NdotL, NdotL, 1);
}
