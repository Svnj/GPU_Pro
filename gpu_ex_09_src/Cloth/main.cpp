
// *** Cloth simulation ***

#include <GL/glew.h>
#include <stdlib.h>
#include <math.h>
#include <GL/freeglut.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "common.h"
#include "Cloth.h"

using namespace std;

// Window size
int width = 600;       
int height = 600;

// camera movement
float center[3];

#define PI 3.141592f

#define ROTATE 1
#define MOVE 2

float thetaStart = PI / 2.0f + 0.1f;
float phiStart = PI / 2.0f + 1.0f;
float rStart = 3.0f;

float theta = thetaStart;
float phi = phiStart;
float r = rStart;

float oldX, oldY;
int motionState;

float viewPosition[3];
float viewDirection[3];

GLuint progSimple;
GLuint uboCamera;
GLuint vboTexCoord;
GLuint iboMesh;
GLuint vao[2];
ClothSim* clothsim;

// Print information about the compiling step
void printShaderInfoLog(GLuint shader)
{
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

void calcViewerCamera(float theta, float phi, float r)
{
    float x = r * sin(theta) * cos(phi);
    float y = r * cos(theta);
    float z = r * sin(theta) * sin(phi);
 
	viewPosition[0] = center[0] + x;
	viewPosition[1] = center[1] + y;
	viewPosition[2] = center[2] + z;
	viewDirection[0] = -x;
	viewDirection[1] = -y;
	viewDirection[2] = -z;

	glLoadIdentity();
	gluLookAt(viewPosition[0], viewPosition[1], viewPosition[2],
				viewPosition[0] + viewDirection[0], viewPosition[1] + viewDirection[1], viewPosition[2] + viewDirection[2], 
				0, 1, 0);

	// update the view matrix.
	float matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	glBindBuffer(GL_UNIFORM_BUFFER, uboCamera);
	glBufferSubData(GL_UNIFORM_BUFFER, sizeof(float) * 16, sizeof(float) * 16, matrix);
}

void mouseMotion(int x, int y)
{
	float deltaX = x - oldX;
	float deltaY = y - oldY;
	
	if (motionState == ROTATE) {
		theta -= 0.001f * deltaY;

		if (theta < 0.001f) theta = 0.001f;
		else if (theta > PI - 0.001f) theta = PI - 0.001f;

		phi += 0.001f * deltaX;	
		if (phi < 0) phi += 2*PI;
		else if (phi > 2*PI) phi -= 2*PI;
		calcViewerCamera(theta, phi, r);
	}
	else if (motionState == MOVE) {
		r += 0.03f * deltaY;
		if (r < 0.1f) r = 0.1f;
		calcViewerCamera(theta, phi, r);
	}

	oldX = (float)x;
	oldY = (float)y;

	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
	oldX = (float)x;
	oldY = (float)y;

	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			motionState = ROTATE;
		}
	}
	else if (button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_DOWN) {
			motionState = MOVE;
		}
	}
}


//------------------------------------------------------------------------
// Rendering loop.
//------------------------------------------------------------------------
void display(void)
{
	// clear frame.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	// bind ubo
	GLuint uniformBlockIndex = glGetUniformBlockIndex(progSimple, "GlobalMatrices");
	glUniformBlockBinding(progSimple, uniformBlockIndex, 0);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, uboCamera, 0, sizeof(float)*32);

	// update gravity system
	clothsim->update(1.0f / 60.0f);

	// get ping pong status
	GLuint ping = clothsim->getPingStatus();
	// setup input assembler
	glBindVertexArray(vao[ping]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboMesh);
	glUseProgram(progSimple);	

	// draw triangles	
	unsigned int indexCount = (RESOLUTION_X - 1) * (RESOLUTION_Y - 1) * 6;
	if (clothsim->getVBOPos(ping) != 0)
		glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);

	// unbind buffer and shader
    glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY); 
	glUseProgram(0);

	// render collision sphere
	glutSolidSphere(SPHERE_RADIUS, 32, 32);

	// swap buffers.    
    glutSwapBuffers();	
}

void idle()
{
	glutPostRedisplay();
}

void initGL()
{
	glClearColor(0,0,0,1);
	// Initialize light source
	GLfloat light_pos[] = {10, 10, 10, 1};
	GLfloat light_col[] = { 1,  1,  1, 1};

	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_col);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_col);

	// Enable lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_DEPTH_TEST);

	// Initialize material
	GLfloat teapot_diffuse[]  = {0.75f, 0.375f, 0.075f, 1};
	GLfloat teapot_specular[] = {0.8f, 0.8f, 0.8f, 1};

	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, teapot_diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, teapot_specular);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 45.2776f);

	// Uniform Buffer Object für die Camera Matrizen anlegen.
	glGenBuffers(1, &uboCamera);
	glBindBuffer(GL_UNIFORM_BUFFER, uboCamera);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 32, NULL, GL_STREAM_DRAW);	

	// Initialize camera
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, 1, 0.1, 100);
	// query projection matrix and update the vbo.
	float matrix[16];
	glGetFloatv(GL_PROJECTION_MATRIX, matrix);
	glBindBuffer(GL_UNIFORM_BUFFER, uboCamera);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(float) * 16, matrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	// Viewmatrix initialisieren
	glMatrixMode(GL_MODELVIEW);
	calcViewerCamera(theta, phi, r);

	// Create index buffer data
	unsigned int indexCount = (RESOLUTION_X - 1) * (RESOLUTION_Y - 1) * 6;
	unsigned int* ib = new unsigned int[indexCount];
	unsigned int i = 0;
	{
		for (unsigned int y=0; y<RESOLUTION_Y - 1; ++y)
		{
			for (unsigned int x=0; x<RESOLUTION_X - 1; ++x)
			{
				unsigned int v_tl = x * RESOLUTION_Y + y;
				unsigned int v_bl = (x + 1) * RESOLUTION_Y + y;
				unsigned int v_tr = x * RESOLUTION_Y + y + 1;
				unsigned int v_br = (x + 1) * RESOLUTION_Y + y + 1;
				ib[i++] = v_bl;
				ib[i++] = v_br;
				ib[i++] = v_tr;
				ib[i++] = v_bl;
				ib[i++] = v_tr;
				ib[i++] = v_tl;				
			}			
		}
	}

	// Create index buffer object
	glGenBuffers(1, &iboMesh);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboMesh);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount*sizeof(unsigned int), ib, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	delete[] ib;

	// Create vertex buffer object for texture coords
	float* texCoords = new float[RESOLUTION_X * RESOLUTION_Y * 2];
	int j=0;
	for (int x=0; x < RESOLUTION_X; ++x)
	{
		for (int y=0; y < RESOLUTION_Y; ++y)
		{
			texCoords[j*2] = x / (float)(RESOLUTION_X-1);
			texCoords[j*2+1] = y / (float)(RESOLUTION_Y-1);
			++j;
		}
	}

	glGenBuffers(1, &vboTexCoord);	
	glBindBuffer(GL_ARRAY_BUFFER, vboTexCoord);
	glBufferData(GL_ARRAY_BUFFER, RESOLUTION_X * RESOLUTION_Y * 2 * sizeof(float), texCoords, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	delete[] texCoords;


	// Create vertex array object
	glGenVertexArrays(2, vao);
	for (int vbo=0; vbo<2; ++vbo)
	{
		glBindVertexArray(vao[vbo]);
			glBindBuffer(GL_ARRAY_BUFFER, clothsim->getVBOPos(vbo));
				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
			glBindBuffer(GL_ARRAY_BUFFER, vboTexCoord);
				glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);		
			// TODO Normalen-VBO von clothsim getten und an das VAO binden.
				glBindBuffer(GL_ARRAY_BUFFER, clothsim->getVBONormal());
				glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);		
			// TODO Vertex Attribut für die Normalen enablen.
			glEnableVertexAttribArray(2);
		glBindVertexArray(0);
	}
}

void initGLSL()
{
	// Create empty shader object (vertex shader)
	GLuint vertexShaderSimple = glCreateShader(GL_VERTEX_SHADER);

	// Read vertex shader source 
	string shaderSource = readFile("Simple.vert");
	const char* sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(vertexShaderSimple, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(vertexShaderSimple);
	printShaderInfoLog(vertexShaderSimple);


	// Create empty shader object (fragment shader)
	GLuint fragmentShaderSimple = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile("Simple.frag");
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(fragmentShaderSimple, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(fragmentShaderSimple);
	printShaderInfoLog(fragmentShaderSimple);

	// Create shader program
	progSimple = glCreateProgram();	

	// Attach shader
	glAttachShader(progSimple, vertexShaderSimple);
	glAttachShader(progSimple, fragmentShaderSimple);

	// Link program
	glLinkProgram(progSimple);
	printProgramInfoLog(progSimple);
}


//------------------------------------------------------------------------
int main(int argc, char** argv)
{
	// Initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("Cloth simulation");

	// Init glew so that the GLSL functionality will be available
	if(glewInit() != GLEW_OK)
	   cout << "GLEW init failed!" << endl;	

	// Select CUDA device for GL interop.
	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof( cudaDeviceProp ) );
	prop.major = 1;
	prop.minor = 0;
	CUDA_SAFE_CALL( cudaChooseDevice(&dev, &prop) );
    cudaGLSetGLDevice( dev );

	// Create cloth simulation.
	clothsim = new ClothSim();

	// OpenGL/GLSL initializations
	initGL();
	initGLSL();	

	// Register callback functions   
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutDisplayFunc(display);
	glutIdleFunc(idle);
	
	// Enter main loop
	glutMainLoop();

	// Delete cloth simulation.
	delete clothsim;

	return 0;
}
