// Framework f�r GLSL-Programme
// Pumping and Glowing Teapot 

#include <GL/glew.h>
#include <stdlib.h>
#include <GL/freeglut.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

// Global variables
GLfloat alpha = 0;

// GLSL related variables
// Blur Shader Program
GLuint vertexShaderBlur = -1;	
GLuint fragmentShaderBlur = -1;
GLuint fragmentShaderBlurVert = -1;
GLuint fragmentShaderBlurHor = -1;

GLuint shaderProgramBlur = -1;
GLuint shaderProgramBlurVert = -1;
GLuint shaderProgramBlurHor = -1;

// Texture Ids and Framebuffer Object Ids
GLuint teapotTextureId = 0;
GLuint depthTextureId = 0;
GLuint blurHorizontalTextureId = 0;

GLuint teapotFB = 0;
GLuint blurHorizontalFB = 0;

// Window size
int width = 512;       
int height = 512;

unsigned int renderMode = 0;

// uniform locations
GLint teapotTextureLocation = -1;
GLint blurHorizontalTextureLocation = -1;
GLint blurVerticalTextureLocation = -1;

bool useBlur = true;

// Print information about the compiling step
void printShaderInfoLog(GLuint shader)
{
	if (shader == -1)
		return;

    GLint infologLength = 0;
    GLsizei charsWritten  = 0;
    char *infoLog;

	glGetShaderiv(shader, GL_INFO_LOG_LENGTH,&infologLength);		
	infoLog = (char *)malloc(infologLength);
	glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
	printf("%s\n",infoLog);
	free(infoLog);
}

// Print information about the linking step
void printProgramInfoLog(GLuint program)
{
	if (program == -1)
		return;

	GLint infoLogLength = 0;
	GLsizei charsWritten  = 0;
	char *infoLog;

	glGetProgramiv(program, GL_INFO_LOG_LENGTH,&infoLogLength);
	infoLog = (char *)malloc(infoLogLength);
	glGetProgramInfoLog(program, infoLogLength, &charsWritten, infoLog);
	printf("%s\n",infoLog);
	free(infoLog);
}

// Reads a file and returns the content as a string
string readFile(string fileName)
{
	string fileContent;
	string line;

	ifstream file(fileName.c_str());
	if (file.is_open()) {
		while (!file.eof()){
			getline (file,line);
			line += "\n";
			fileContent += line;					
		}
		file.close();
	}
	else
		cout << "ERROR: Unable to open file " << fileName << endl;

	return fileContent;
}

void initGL()
{
   // Initialize camera
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(45, 1, 0.1, 100);
   glMatrixMode(GL_MODELVIEW);

   // Initialize light source
   GLfloat light_pos[] = {10, 10, 10, 1};
   GLfloat light_col[] = { 1,  1,  1, 1};

   glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
   glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_col);
   glLightfv(GL_LIGHT0, GL_SPECULAR, light_col);

   // Enable lighting
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);

   // Initialize material
   GLfloat teapot_diffuse[]  = {0.75f, 0.375f, 0.075f, 1};
   GLfloat teapot_specular[] = {0.8f, 0.8f, 0.8f, 1};

   glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, teapot_diffuse);
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, teapot_specular);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 45.2776f);

   // Enable depth buffer
   glEnable(GL_DEPTH_TEST);
}

void initGLSL()
{
	// TODO: Create empty shader object (vertex shader) and assign it to 'vertexShaderBlur'
	vertexShaderBlur = glCreateShader(GL_VERTEX_SHADER);

	// Read vertex shader source 
	string shaderSource = readFile("blur.vert");
	const char* sourcePtr = shaderSource.c_str();

	// TODO: Attach shader code
	glShaderSource (vertexShaderBlur, 1, &sourcePtr, NULL);

	// TODO: Compile shader	
	glCompileShader (vertexShaderBlur);
	
	printShaderInfoLog(vertexShaderBlur);

	// TODO: Create empty shader object (fragment shader) and assign it to 'fragmentShaderBlur'
	fragmentShaderBlur = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile("blur.frag");
	sourcePtr = shaderSource.c_str();

	// TODO: Attach shader code
	glShaderSource (fragmentShaderBlur, 1, &sourcePtr, NULL);

	// TODO: Compile shader
	glCompileShader (fragmentShaderBlur);	

	printShaderInfoLog(fragmentShaderBlur);

	// TODO: Create empty shader object (fragment shader) and assign it to 'fragmentShaderBlurVert'
	fragmentShaderBlurVert = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile("blur_vert.frag");
	sourcePtr = shaderSource.c_str();

	// TODO: Attach shader code
	glShaderSource (fragmentShaderBlurVert, 1, &sourcePtr, NULL);

	// TODO: Compile shader
	glCompileShader (fragmentShaderBlurVert);	

	printShaderInfoLog(fragmentShaderBlurVert);

	// TODO: Create empty shader object (fragment shader) and assign it to 'fragmentShaderBlurHor'
	fragmentShaderBlurHor = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile("blur_hor.frag");
	sourcePtr = shaderSource.c_str();

	// TODO: Attach shader code
	glShaderSource (fragmentShaderBlurHor, 1, &sourcePtr, NULL);

	// TODO: Compile shader
	glCompileShader (fragmentShaderBlurHor);	

	printShaderInfoLog(fragmentShaderBlurHor);

	// TODO: Create shader program and assign it to 'shaderProgramBlur'
	shaderProgramBlur = glCreateProgram ();	

	// TODO: Attach shader vertex shader and fragment shader to program	
    glAttachShader(shaderProgramBlur, fragmentShaderBlur);
    glAttachShader(shaderProgramBlur, vertexShaderBlur);

	// TODO: Link program
    glLinkProgram(shaderProgramBlur);
	
	printProgramInfoLog(shaderProgramBlur);

	// TODO: Create shader program and assign it to 'shaderProgramBlurVert'
	shaderProgramBlurVert = glCreateProgram ();	

	// TODO: Attach shader vertex shader and fragment shader to program	
    glAttachShader(shaderProgramBlurVert, fragmentShaderBlurVert);
    glAttachShader(shaderProgramBlurVert, vertexShaderBlur);

	// TODO: Link program
    glLinkProgram(shaderProgramBlurVert);
	
	printProgramInfoLog(shaderProgramBlurVert);

	// TODO: Create shader program and assign it to 'shaderProgramBlurHor'
	shaderProgramBlurHor = glCreateProgram ();	

	// TODO: Attach shader vertex shader and fragment shader to program	
    glAttachShader(shaderProgramBlurHor, fragmentShaderBlurHor);
    glAttachShader(shaderProgramBlurHor, vertexShaderBlur);

	// TODO: Link program
    glLinkProgram(shaderProgramBlurHor);
	
	printProgramInfoLog(shaderProgramBlurHor);

	// TODO: Use program.	

	// Eingabe in diesen Shader ist die Textur, in die die Szene gerendert wird.
	// An dieser Stelle wird die uniform Location f�r die Textur-Variable im Shader geholt.
	/* TextureLocations:
	- teapotTextureLocation;
	- blurHorizontalTextureLocation;
	- blurVerticalTextureLocation;
	*/

	glUseProgram(shaderProgramBlurVert);
	blurVerticalTextureLocation = glGetUniformLocation( shaderProgramBlurVert, "texture" );
	glUniform1i(blurVerticalTextureLocation, 1);   
	if(blurVerticalTextureLocation == -1)
		cout << "ERROR: No uniform texture in shaderProgramBlurVert" << endl;
	
    glUseProgram(shaderProgramBlurHor);
	blurHorizontalTextureLocation = glGetUniformLocation( shaderProgramBlurHor, "texture" );
	glUniform1i(blurHorizontalTextureLocation, 0);   
	if(teapotTextureLocation == -1)
		cout << "ERROR: No uniform texture in shaderProgramBlurHor" << endl;	

	glUseProgram(shaderProgramBlur);
	teapotTextureLocation = glGetUniformLocation( shaderProgramBlur, "texture" );
	glUniform1i(teapotTextureLocation, 0);   
	if(teapotTextureLocation == -1)
		cout << "ERROR: No uniform texture in shaderProgramBlur" << endl;

	//glUseProgram(shaderProgramBlur);

}


int initFBOTextures()
{
	// Textur (fuer Teapot Bild) anlegen
	glGenTextures (1, &teapotTextureId);
	glBindTexture (GL_TEXTURE_2D, teapotTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// Textur (fuer horizontalen Filter) anlegen
	glGenTextures (1, &blurHorizontalTextureId);
	glBindTexture (GL_TEXTURE_2D, blurHorizontalTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// Depth Buffer Textur anlegen 
	glGenTextures (1, &depthTextureId);
	glBindTexture (GL_TEXTURE_2D, depthTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// FBO (fuer Teapot Textur) anlegen und Texturen zuweisen
	glGenFramebuffers (1, &teapotFB);
	glBindFramebuffer (GL_FRAMEBUFFER, teapotFB);
	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, teapotTextureId, 0);
	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureId, 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture (GL_TEXTURE_2D, teapotTextureId); // texture 0 is the teapot color buffer
	
	// FBO (fuer horizontalen Filter) anlegen und Texturen zuweisen
	glGenFramebuffers (1, &blurHorizontalFB);
	glBindFramebuffer (GL_FRAMEBUFFER, blurHorizontalFB);
	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, blurHorizontalTextureId, 0);
	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureId, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture (GL_TEXTURE_2D, blurHorizontalTextureId); // texture 1 is the horizontal filter color buffer
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture (GL_TEXTURE_2D, teapotTextureId);

	// check framebuffer status
	GLenum status = glCheckFramebufferStatus (GL_FRAMEBUFFER);
	switch (status)
	{
	case GL_FRAMEBUFFER_COMPLETE:
		cout << "FBO complete" << endl;
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED:
		cout << "FBO configuration unsupported" << endl;
		return 1;
	default:
		cout << "FBO programmer error" << endl;
		return 1;
	}
	glBindFramebufferEXT (GL_FRAMEBUFFER, 0);
	return 0;
}

void keyboard(unsigned char key, int x, int y)
{
	// set parameters
	switch (key) 
	{       
		case ' ':
			renderMode = (renderMode+1)%3;
			break;
	}
}

// Bildschirmfuellendes Rechteck zeichnen -> Fragment Program wird fuer jedes Pixel aufgerufen
void drawScreenFillingQuad() 
{
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0,0);
		glVertex2f(-1,-1);
		glTexCoord2f(1,0);
		glVertex2f( 1,-1);
		glTexCoord2f(1,1);
		glVertex2f(1,1);
		glTexCoord2f(0,1);
		glVertex2f( -1,1);
	}       
	glEnd();

	glPopMatrix();	
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
} 

void display()
{	
	/* 	Textures:
		- teapotTextureId;	
		- depthTextureId;
		- blurHorizontalTextureId

		FBO's:
		- teapotFB
		- blurHorizontalFB;
	*/

	int timeStart = glutGet(GLUT_ELAPSED_TIME);

	// Pumping Shader anschalten falls aktiviert
	glUseProgram( 0 );
	
	// falls Blur Shader aktiviert ist, muss in eine Textur gerendert werden
	if (renderMode == 1 | renderMode == 2)
		glBindFramebuffer (GL_FRAMEBUFFER, teapotFB);      // activate fbo

	// Clear window
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();
	gluLookAt(10, 7, 10, 0, 0, 0, 0, 1, 0);

	glRotatef(alpha, 0, 1, 0);
	glutSolidTeapot(3);

	// FBO abschalten: jetzt wird wieder in den Framebuffer gerendert
	glBindFramebuffer (GL_FRAMEBUFFER, 0);      // deactivate fbo

	// Blur Shader aktivieren und bildschirmfuellendes Rechteck zeichnen
	if (renderMode == 1) {
		
		glUseProgram(shaderProgramBlur); // activate blur shader

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawScreenFillingQuad();

		glUseProgram( 0 );

		// TODO: Teapot oben dr�ber zeichnen.
		glutSolidTeapot(3);
	}

	if(renderMode == 2)
	{	
		glBindFramebuffer (GL_FRAMEBUFFER, blurHorizontalFB);
		glUseProgram(shaderProgramBlurHor); // activate blur shader		

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawScreenFillingQuad();
		

		glBindFramebuffer (GL_FRAMEBUFFER, 0);
		glUseProgram( shaderProgramBlurVert );

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawScreenFillingQuad();		

		glUseProgram( 0 );

		// TODO: Teapot oben dr�ber zeichnen.
		glutSolidTeapot(3);
	}

	// Increment rotation angle
	alpha += 1;

	// Swap display buffers
	glutSwapBuffers();

	int timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("rendermode: %d  ,Delay %d     \r",renderMode,timeEnd - timeStart);
}

void timer(int value)
{
   // Call timer() again in 25 milliseconds
   glutTimerFunc(25, timer, 0);

   // Redisplay frame
   glutPostRedisplay();
}

int main(int argc, char** argv)
{
   // Initialize GLUT
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
   glutInitWindowSize(width, height);
   glutCreateWindow("Glowing Teapot");

   // Init glew so that the GLSL functionality will be available
   if(glewInit() != GLEW_OK)
	   cout << "GLEW init failed!" << endl;

	// OpenGL/GLSL initializations
	initGL();
	initFBOTextures();
	initGLSL();

	// Register callback functions   
	glutKeyboardFunc(keyboard);
	glutDisplayFunc(display);
	glutTimerFunc(25, timer, 0);     // Call timer() in 25 milliseconds

	// Enter main loop
	glutMainLoop();

	return 0;
}
