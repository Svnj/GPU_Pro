uniform sampler2D imageTexture;

const float SCREEN_WIDTH = 512.0;

void main()
{	
	// TODO: Farbe auslesen
	vec2 TexCoord = gl_Vertex.xy / SCREEN_WIDTH;
	vec3 color = texture2D(imageTexture, TexCoord.st).rgb;
 	
	// TODO: Grauwert berechnen
	float greyscale = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;

//	// TODO: x-Position berechnen. Das Zielpixel ist zwischen (0,0) und (255,0)
//	// TODO: Die Position in [0,1] auf das Intervall [-1,1] abbilden.
        gl_Position = vec4(-1.0f + greyscale, -1.0f, 0.0f, 1.0f);
}
