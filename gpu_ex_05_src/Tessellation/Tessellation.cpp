
// *** Tessellation Demo ***

#include <GL/glew.h>
#include <stdlib.h>
#include <math.h>
#include <GL/freeglut.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

// Window size
int width = 800;       
int height = 800;

// camera movement
float center[3] = {0, 350, 0};

#define PI 3.141592f

#define ROTATE 1
#define MOVE 2

float thetaStart = PI / 2.0f - 0.5f;
float phiStart = PI / 2.0f;
float rStart = 1000.0f;

float theta = thetaStart;
float phi = phiStart;
float r = rStart;

float oldX, oldY;
int motionState;

float viewPosition[3];
float viewDirection[3];

GLuint vaoelephant;
GLuint iboelephant;
GLuint progTessellation;
GLuint uboCamera;
GLuint uboTessellation;

GLfloat insideTess = 4;
GLfloat outsideTess = 4;
GLfloat alpha = 1;
bool wireframe = true;

extern float elephantData[];
extern unsigned int elephantStride;
extern unsigned int elephantSize;
extern unsigned int elephantIndices[];
extern unsigned int elephantIndicesStride;
extern unsigned int elephantIndicesSize;

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
		r += 0.7f * deltaY;
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

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
		case 27:
			exit(0);	
			break;
		case '1': // inside Tessellation Factor
			insideTess = insideTess - 0.1f;	insideTess = insideTess < 1 ? 1 : insideTess;	insideTess = insideTess > 64 ? 64 : insideTess;
			glBindBuffer(GL_UNIFORM_BUFFER, uboTessellation);
			glBufferSubData(GL_UNIFORM_BUFFER, 0, 4, &insideTess);
			glBindBuffer(GL_UNIFORM_BUFFER, 0);
			printf("insideTess: %.f ,  outside: %.f ,  alpha: %.2f ,  wireframe: %s      \r", insideTess, outsideTess, alpha, wireframe?"true":"false");
			break;
		case '2': // inside Tessellation Factor
			insideTess = insideTess + 0.1f;	insideTess = insideTess < 1 ? 1 : insideTess;	insideTess = insideTess > 64 ? 64 : insideTess;
			glBindBuffer(GL_UNIFORM_BUFFER, uboTessellation);
			glBufferSubData(GL_UNIFORM_BUFFER, 0, 4, &insideTess);
			glBindBuffer(GL_UNIFORM_BUFFER, 0);
			printf("insideTess: %.f ,  outside: %.f ,  alpha: %.2f ,  wireframe: %s      \r", insideTess, outsideTess, alpha, wireframe?"true":"false");
			break;

		case '3': // outside Tessellation Factor
			outsideTess = outsideTess - 0.1f;	outsideTess = outsideTess < 1 ? 1 : outsideTess;	outsideTess = outsideTess > 64 ? 64 : outsideTess;
			glBindBuffer(GL_UNIFORM_BUFFER, uboTessellation);
			glBufferSubData(GL_UNIFORM_BUFFER, 4, 4, &outsideTess);
			glBindBuffer(GL_UNIFORM_BUFFER, 0);
			printf("insideTess: %.f ,  outside: %.f ,  alpha: %.2f ,  wireframe: %s      \r", insideTess, outsideTess, alpha, wireframe?"true":"false");
			break;
		case '4': // outside Tessellation Factor
			outsideTess = outsideTess + 0.1f;	outsideTess = outsideTess < 1 ? 1 : outsideTess;	outsideTess = outsideTess > 64 ? 64 : outsideTess;
			glBindBuffer(GL_UNIFORM_BUFFER, uboTessellation);
			glBufferSubData(GL_UNIFORM_BUFFER, 4, 4, &outsideTess);
			glBindBuffer(GL_UNIFORM_BUFFER, 0);
			printf("insideTess: %.f ,  outside: %.f ,  alpha: %.2f ,  wireframe: %s      \r", insideTess, outsideTess, alpha, wireframe?"true":"false");
			break;
		case '+':
			alpha += 0.01f;
			glBindBuffer(GL_UNIFORM_BUFFER, uboTessellation);
			glBufferSubData(GL_UNIFORM_BUFFER, 8, 4, &alpha);
			glBindBuffer(GL_UNIFORM_BUFFER, 0);
			printf("insideTess: %.f ,  outside: %.f ,  alpha: %.2f ,  wireframe: %s      \r", insideTess, outsideTess, alpha, wireframe?"true":"false");
			break;
		case '-':
			alpha -= 0.01f;
			glBindBuffer(GL_UNIFORM_BUFFER, uboTessellation);
			glBufferSubData(GL_UNIFORM_BUFFER, 8, 4, &alpha);
			glBindBuffer(GL_UNIFORM_BUFFER, 0);
			printf("insideTess: %.f ,  outside: %.f ,  alpha: %.2f ,  wireframe: %s      \r", insideTess, outsideTess, alpha, wireframe?"true":"false");
			break;
		case 'w':
			wireframe = !wireframe;
			printf("insideTess: %.f ,  outside: %.f ,  alpha: %.2f ,  wireframe: %s      \r", insideTess, outsideTess, alpha, wireframe?"true":"false");
			break;
	}
	glutPostRedisplay();
}

//------------------------------------------------------------------------
// Rendering loop.
//------------------------------------------------------------------------
void display(void)
{
	// clear frame.
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GLuint uniformBlockIndex = glGetUniformBlockIndex(progTessellation, "GlobalMatrices");
	glUniformBlockBinding(progTessellation, uniformBlockIndex, 0);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, uboCamera, 0, sizeof(float)*32);

	uniformBlockIndex = glGetUniformBlockIndex(progTessellation, "TessFactors");
	glUniformBlockBinding(progTessellation, uniformBlockIndex, 1);
	glBindBufferRange(GL_UNIFORM_BUFFER, 1, uboTessellation, 0, sizeof(float)*3);		

	// polygon fill mode
	glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

	// Bind VAO and IBO
	glBindVertexArray(vaoelephant);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboelephant);

	// TODO: Shader binden
    glUseProgram(progTessellation);

	// TODO: GL mitteilen, dass ein Patch aus drei Vertices besteht
    glPatchParameteri (GL_PATCH_VERTICES, 3);
	
	// TODO: Primitiv-Typ auf Patches umstellen.
    glDrawElements(GL_PATCHES, elephantIndicesSize/elephantIndicesStride, GL_UNSIGNED_INT, 0);
	glUseProgram(0);
	
	// Unbind VAO and IBO
	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// Flush command buffer and swap buffers.
	glFlush();
	glutSwapBuffers();
}


void initelephant()
{
	// Create vertex buffer object
	GLuint vbo;	
	glGenBuffers(1, &vbo);	
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, elephantSize, elephantData, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Create index buffer object
	glGenBuffers(1, &iboelephant);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboelephant);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, elephantIndicesSize, elephantIndices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// Create vertex array object
	glGenVertexArrays(1, &vaoelephant);
	glBindVertexArray(vaoelephant);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, elephantStride, 0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, elephantStride, (char*)12);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, elephantStride, (char*)24);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
	glBindVertexArray(0);
}

void initGL()
{
	glClearColor(1,1,1,1);
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
	GLfloat elephant_diffuse[]  = {0.75f, 0.375f, 0.075f, 1};
	GLfloat elephant_specular[] = {0.8f, 0.8f, 0.8f, 1};

	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, elephant_diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, elephant_specular);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 45.2776f);

	// Uniform Buffer Object fuer die Camera Matrizen anlegen.
	glGenBuffers(1, &uboCamera);
	glBindBuffer(GL_UNIFORM_BUFFER, uboCamera);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 32, NULL, GL_STREAM_DRAW);	

	// Initialize camera
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, 1, 0.1, 10000);
	// query projection matrix and update the vbo.
	float matrix[16];
	glGetFloatv(GL_PROJECTION_MATRIX, matrix);
	glBindBuffer(GL_UNIFORM_BUFFER, uboCamera);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(float) * 16, matrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	// Viewmatrix initialisieren
	glMatrixMode(GL_MODELVIEW);
	calcViewerCamera(theta, phi, r);

	// Enable depth buffer
	glEnable(GL_DEPTH_TEST);


	glGenBuffers(1, &uboTessellation);
	glBindBuffer(GL_UNIFORM_BUFFER, uboTessellation);
	float init[3] = { insideTess, outsideTess, alpha };
	glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 3, init, GL_STREAM_DRAW);	
}

void initGLSL()
{
	// ---------------------------------------------------
	// Create empty shader object (vertex shader)
	GLuint vertexShaderSimple = glCreateShader(GL_VERTEX_SHADER);

	// Read vertex shader source 
    string shaderSource = readFile("Tessellation.vert");
	const char* sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(vertexShaderSimple, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(vertexShaderSimple);
	printShaderInfoLog(vertexShaderSimple);

	// ---------------------------------------------------
	// Create empty shader object (hull shader)
	GLuint hullShaderSimple = glCreateShader(GL_TESS_CONTROL_SHADER);

	// Read vertex shader source 
    shaderSource = readFile("Tessellation.hull");
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(hullShaderSimple, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(hullShaderSimple);
	printShaderInfoLog(hullShaderSimple);

	// ---------------------------------------------------
	// Create empty shader object (domain shader)
	GLuint domainShaderSimple = glCreateShader(GL_TESS_EVALUATION_SHADER);

	// Read vertex shader source 
    shaderSource = readFile("Tessellation.dom");
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(domainShaderSimple, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(domainShaderSimple);
	printShaderInfoLog(domainShaderSimple);

	// ---------------------------------------------------
	// Create empty shader object (fragment shader)
	GLuint fragmentShaderSimple = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
    shaderSource = readFile("Tessellation.frag");
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(fragmentShaderSimple, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(fragmentShaderSimple);
	printShaderInfoLog(fragmentShaderSimple);

	// ---------------------------------------------------
	// Create shader program
	progTessellation = glCreateProgram();	

	// Attach shader
	glAttachShader(progTessellation, vertexShaderSimple);
    glAttachShader(progTessellation, hullShaderSimple);
    glAttachShader(progTessellation, domainShaderSimple);
    glAttachShader(progTessellation, fragmentShaderSimple);
	
    // Link program
	glLinkProgram(progTessellation);
    printProgramInfoLog(progTessellation);
}

//------------------------------------------------------------------------
//   It's the main application function. Note the clean code you can
//   obtain using he GLUT library. No calls to dark windows API
//   functions with many obscure parameters list. =)
//------------------------------------------------------------------------
int main(int argc, char** argv)
{
	 // Initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("Hairy elephant");

	// Init glew so that the GLSL functionality will be available
	if(glewInit() != GLEW_OK)
		cout << "GLEW init failed!" << endl;
	
	cout << "Keys:" << endl;
	cout << "  Modifiy  inside tessellation factors: (1), (2)" << endl;
	cout << "  Modifiy outside tessellation factors: (3), (4)" << endl;
	cout << "  Modifiy alpha: (+), (-)" << endl;
	cout << "  Wireframe on/off: (w)" << endl;

	// OpenGL/GLSL initializations
	initGL();
	initelephant();
	initGLSL();

	// Register callback functions   
	glutKeyboardFunc(keyboard);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutDisplayFunc(display);
	
	// Enter main loop
	glutMainLoop();

	return 0;
}
