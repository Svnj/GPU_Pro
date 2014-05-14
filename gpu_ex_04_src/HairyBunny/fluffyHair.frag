// simple fragment shader that outputs transparent white (as hair color)

#version 150

flat in vec3 originatingVertex;
in vec3 geomNormalInEyeSpace;
in vec3 geomPositionInEyeSpace;

out vec4 fragColor;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};


// fixed point light properties
vec3 light_position_world  = vec3 (0.0, 0.0, 2.0);
vec3 Ls = vec3 (1.0, 1.0, 1.0); // white specular colour
vec3 Ld = vec3 (0.7, 0.7, 0.7); // dull white diffuse light colour
vec3 La = vec3 (0.2f); // grey ambient colour

// surface reflectance
vec3 Ks = vec3 (1.0, 1.0, 1.0); // fully reflect specular light
vec3 Kd;
vec3 Ka = vec3 (0.5); // fully reflect ambient light
float specular_exponent = 100.0; // specular 'power'

float rand(vec2 seed){
    return fract(sin(dot(seed.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{	
	// here i try to get some random brownish colours
	// i use the vertex from which the hair strand originated as the seed 
	// because i dont want the hair strand to change colour when i change the view 	
	Kd.r = (130 + (20 * rand(originatingVertex.xy)))/255;
	Kd.g = (65 + (10 * rand(originatingVertex.xz)))/255;
	Kd.b = (10 + (20 * rand(originatingVertex.yz)))/255;

  // ambient intensity
  vec3 Ia = La * Ka;


  // diffuse intensity
  // raise light position to eye space
  vec3 light_position_eye = vec3 (View * vec4 (light_position_world, 1.0));
  vec3 distance_to_light_eye = light_position_eye - geomPositionInEyeSpace;
  vec3 direction_to_light_eye = normalize (distance_to_light_eye);
  float dot_prod = dot (direction_to_light_eye, geomNormalInEyeSpace);
  dot_prod = max (dot_prod, 0.0);
  vec3 Id = Ld * Kd * dot_prod; // final diffuse intensity

  // specular intensity
  vec3 reflection_eye = reflect (-direction_to_light_eye, geomNormalInEyeSpace);
  vec3 surface_to_viewer_eye = normalize (-geomPositionInEyeSpace);
  float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
  dot_prod_specular = max (dot_prod_specular, 0.0);
  float specular_factor = pow (dot_prod_specular, specular_exponent);
  vec3 Is = Ls * Ks * specular_factor; // final specular intensity

  // final colour
  fragColor = vec4 ( Ia + Id + Is, 1.0);
}
