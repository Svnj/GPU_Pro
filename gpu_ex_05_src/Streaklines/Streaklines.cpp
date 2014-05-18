
// *** Streakline visualization with transform feedback. ***

#include <GL/glew.h>
#include "GL/freeglut.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

GLuint progAdvect;				// Shader for the advection and rendering of the particles.
GLuint texFlow;					// Texture containing the vector field data.

GLuint vbo_streamTo = 0;
GLuint vao_streamTo = 0;
GLuint feedback_streamTo = 0;
GLuint vbo_readFrom = 0;
GLuint vao_readFrom = 0;
GLuint feedback_readFrom = 0;
GLboolean bFirst = true;		// first draw call? (if yes, we can't yet use glDrawTransformFeedback, since its counter is still 0.)
GLuint Query = 0;				// Query object used to retrieve the number particles stream out to the "streamTo" buffer. (Only used for debugging purposes.)

// ------- Uniform Buffer Objects --------
GLuint ubo_Camera;
GLuint ubo_Params;

// Simple helper for swapping to variables.
void swap(GLuint& A, GLuint& B)
{
	GLuint temp = A;
	A = B;
	B = temp;
}

// ------- Particle --------
struct PARTICLE_VERTEX
{
    GLfloat pos[2];	// 2D position of the particle.
	GLuint state;	// // 2=Head, 1=Body, 0=Tail
};
#define MAX_PARTICLES 1000000

#define NUM_SEEDS 250


float box_min[3] = {0};
float box_max[3] = {0};
int dim[3] = {0};
float timeCount = 0;
float stepSize = 0.004f;

float seedlineA[2] = {-0.4f,-0.45f};
float seedlineB[2] = {-0.4f, 0.45f};

// Loads vector field data (defined in amira)
extern float* LoadField(const char* FileName, 
							int* xDim, int* yDim, int* zDim, 
							float* xmin, float* ymin, float* zmin, 
							float* xmax, float* ymax, float* zmax);

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

//------------------------------------------------------------------------
// Rendering loop.
//------------------------------------------------------------------------
void display(void)
{
	// clear frame.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Bind matrices
	GLuint uniformBlockIndex = glGetUniformBlockIndex(progAdvect, "GlobalMatrices");
	glUniformBlockBinding(progAdvect, uniformBlockIndex, 0);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo_Camera, 0, sizeof(float)*16);

	// update time and step size
	glBindBuffer(GL_UNIFORM_BUFFER, ubo_Params);
	glBufferSubData(GL_UNIFORM_BUFFER, sizeof(float)*4, sizeof(float), &stepSize);
    glBufferSubData(GL_UNIFORM_BUFFER, sizeof(float)*5, sizeof(float), &timeCount);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	// Bind parameters
	uniformBlockIndex = glGetUniformBlockIndex(progAdvect, "Params");
	glUniformBlockBinding(progAdvect, uniformBlockIndex, 1);
	glBindBufferRange(GL_UNIFORM_BUFFER, 1, ubo_Params, 0, sizeof(float)*6);

	// Bind shader
	glUseProgram(progAdvect);

	// Bind flow texture
	glBindTexture(GL_TEXTURE_3D, texFlow);
	GLuint loc = glGetUniformLocation(progAdvect, "texFlow");
	glUniform1i(loc, 0);

	// Starte query (nur fuer Debugging gedacht!)
	glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, Query); 

	// TODO: Binden des Feedback Objekts (feedback_streamTo)
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,feedback_streamTo);

	// TODO: Beginne Feedback Recording von GL_POINTS
    glBeginTransformFeedback(GL_POINTS);
	// TODO: Binden des VAO von dem gelesen werden soll
    glBindVertexArray(vao_readFrom);
	
	// In der ersten Iteration glDrawArrays verwenden, spaeter den Feedback Draw Call nehmen.
	if( bFirst ) {
        glDrawArrays(GL_POINTS, 0, NUM_SEEDS*2);
	}
	else {
		// TODO: Den Transform Feedback Draw-Call nehmen, da dieser bereits weiss, wieviele Vertices sich derzeit im Stream befinden.
		// D.h. wir muessen diese Zahl nicht per Query zuruecklesen, um den Draw-Call abzusetzen.
        glDrawTransformFeedback(GL_POINTS,feedback_readFrom);
	}	

	// TODO: Beenden des Feedback Recording
    glEndTransformFeedback();

	// End query
	glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN); 

	// Unbind stuff.
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
	glUseProgram(0);
	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_3D, 0);

	// Read back query result.
	GLuint PrimitivesWritten = 0;
	glGetQueryObjectuiv(Query, GL_QUERY_RESULT, &PrimitivesWritten);
	printf("Written: %i \r", PrimitivesWritten);

	// Swap buffers.
    glutSwapBuffers();

	// Swap particle buffers (ping-pong)
	swap(vao_readFrom, vao_streamTo);
	swap(vbo_readFrom, vbo_streamTo);
	swap(feedback_readFrom, feedback_streamTo);

	bFirst = false;

	// Let the time progress.
    timeCount += stepSize / (box_max[2]-box_min[2]);
}


void initGL()
{
	glClearColor(0,0,0,1);
	
	// Enable depth buffer
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_3D);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation(GL_ADD);

	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	glOrtho (box_min[0], box_max[0], box_min[1], box_max[1], 0, 1);


	// Uniform Buffer Object fuer die Camera Matrix anlegen.
	glGenBuffers(1, &ubo_Camera);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo_Camera);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 16, NULL, GL_STREAM_DRAW);	

	// query projection matrix and update the ubo.
	float matrix[16];
	glGetFloatv(GL_PROJECTION_MATRIX, matrix);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo_Camera);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(float) * 16, matrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);


	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity ();
}

void initGLSL()
{
	// Create empty shader object (vertex shader)
	GLuint vertexShaderAdvect = glCreateShader(GL_VERTEX_SHADER);

	// Read vertex shader source 
	string shaderSource = readFile("advect.vert");
	const char* sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(vertexShaderAdvect, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(vertexShaderAdvect);
	printShaderInfoLog(vertexShaderAdvect);

	// Create empty shader object (geometry shader)
	GLuint geometryShaderAdvect = glCreateShader(GL_GEOMETRY_SHADER);

	// Read vertex shader source 
	shaderSource = readFile("advect.geom");
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(geometryShaderAdvect, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(geometryShaderAdvect);
	printShaderInfoLog(geometryShaderAdvect);

	// Create empty shader object (fragment shader)
	GLuint fragmentShaderAdvect = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile("advect.frag");
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(fragmentShaderAdvect, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(fragmentShaderAdvect);
	printShaderInfoLog(fragmentShaderAdvect);

	// Create shader program
	progAdvect = glCreateProgram();	

	// Attach shader
	glAttachShader(progAdvect, vertexShaderAdvect);
	glAttachShader(progAdvect, geometryShaderAdvect);
	glAttachShader(progAdvect, fragmentShaderAdvect);
	
	// Output definition (glTransformFeedbackVaryings)
	GLchar const * Strings[] = {"gs_out_Position", "gs_out_State"}; 
	glTransformFeedbackVaryings(progAdvect, 2, Strings, GL_INTERLEAVED_ATTRIBS);

	// Link program
	glLinkProgram(progAdvect);
	printProgramInfoLog(progAdvect);
}

void initFlow(float* flowData)
{
	// Erzeugen eines Texturnames (handle).
	glGenTextures(1, &texFlow);

	// Binden der Textur.
	glBindTexture(GL_TEXTURE_3D, texFlow);

	// Fuellen der Textur.
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, dim[0], dim[1], dim[2], 0, GL_RG, GL_FLOAT, flowData);

	// Textur filter auf trilineare interpolation.
	glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);	

	// Die Textur muss nicht mehr gebunden sein.
	glBindTexture(GL_TEXTURE_3D, 0);
	
	// Die Daten aus dem Hauptspeicher koennen weg, da sie nun im Video RAM liegen.
	if (flowData) free(flowData);
}

#ifndef BUFFER_OFFSET
#define BUFFER_OFFSET(i) ((char*)NULL + (i))
#endif

void initArrayBuffer()
{
	// Generate initial seeds
	PARTICLE_VERTEX* vertStart = new PARTICLE_VERTEX[MAX_PARTICLES];
	for (int i=0; i<NUM_SEEDS; ++i)
    {
		float t = i / (float)(NUM_SEEDS-1);
		vertStart[i*2 + 0].pos[0] = seedlineA[0] * (1-t) + seedlineB[0] * t;
		vertStart[i*2 + 0].pos[1] = seedlineA[1] * (1-t) + seedlineB[1] * t;
		vertStart[i*2 + 0].state = 2;	// Head

		vertStart[i*2 + 1].pos[0] = vertStart[i*2 + 0].pos[0];
		vertStart[i*2 + 1].pos[1] = vertStart[i*2 + 0].pos[1];
		vertStart[i*2 + 1].state = 0;	// Tail
    }
	
	glGenBuffers(1, &vbo_streamTo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_streamTo);
	glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof( PARTICLE_VERTEX ), vertStart, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_readFrom);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_readFrom);
	glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof( PARTICLE_VERTEX ), vertStart, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	delete [] vertStart;
}

void initVertexArray()
{
	const int stride = sizeof(PARTICLE_VERTEX);

	// VAOs (vao_streamTo und  vao_readFrom) bauen.

	//  Jeder Vertex enthaelt zwei Partikel. Es werden also 4 Attribute reingereicht:
	//    PositionA (offset = 0)
	//    StateA    (offset = 8 = sizeof(float)*2)
	//    PositionB (offset = 12 = stride)
	//    StateB    (offset = 20 = stride + 8)

	// Um jedes Partikel einzeln mit seinem Nachfolger zu betrachten, muss der stride 12 (=sizeof(PARTICLE_VERTEX)) sein! 

	// Build a vertex array object
	glGenVertexArrays(1, &vao_streamTo);
	glBindVertexArray(vao_streamTo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_streamTo);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, BUFFER_OFFSET(0));		// PositionA
		glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, stride, BUFFER_OFFSET(8));		// StateA

		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, BUFFER_OFFSET(stride+0));	// PositionB
		glVertexAttribIPointer(3, 1, GL_UNSIGNED_INT, stride, BUFFER_OFFSET(stride+8));		// StateB

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glEnableVertexAttribArray(3);
	glBindVertexArray(0);

	// Build a vertex array object
	glGenVertexArrays(1, &vao_readFrom);
	glBindVertexArray(vao_readFrom);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_readFrom);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, BUFFER_OFFSET(0));		// PositionA
		glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, stride, BUFFER_OFFSET(8));		// StateA

		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, BUFFER_OFFSET(stride+0));	// PositionB
		glVertexAttribIPointer(3, 1, GL_UNSIGNED_INT, stride, BUFFER_OFFSET(stride+8));		// StateB

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glEnableVertexAttribArray(3);
	glBindVertexArray(0);
}

void initFeedback()
{
	// TODO: Buffer Objekt generieren und in Variable feedback_readFrom speichern.
    glGenTransformFeedbacks(1,&feedback_readFrom);
	// TODO: Buffer Objekt binden
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,feedback_readFrom);
	// TODO: vbo_readFrom mit dem Buffer Objekt verknuepfen (wenn man das Transform Feedback Objekt bindet, wird kuenftig in dieses VBO geschrieben)
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER,0,vbo_readFrom);
	// TODO: Buffer Objekt unbinden
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,0);

	// TODO: Das selbe nochmal fuer feedback_streamTo
    glGenTransformFeedbacks(1,&feedback_streamTo);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,feedback_streamTo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER,0,vbo_streamTo);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,0);
}

void initParams()
{
	glGenBuffers(1, &ubo_Params);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo_Params);

	float initial[] = {
		box_min[0], box_max[0],		// xRange
		box_min[1], box_max[1],		// yRange
		stepSize,					// stepSize
        timeCount						// time
	};
	glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 6, initial, GL_STREAM_DRAW);		
}

void idle()
{
	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	// Load vector field (raw data, resolution of the grid and the bounding box).
    const char* FileName = "Cylinder2D.am";
	float* flowData = LoadField(FileName, &dim[0], &dim[1], &dim[2], &box_min[0], &box_min[1], &box_min[2], &box_max[0], &box_max[1], &box_max[2]);

	if (!flowData)
	{
		cout << "Couldn't find Cylinder2D.am." << endl;
		cout << "You can download it from http://www.courant.nyu.edu/~weinkauf/datasets/Cylinder2D.7z" << endl;
	}

	// Initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(dim[0]*3, dim[1]*3);
	glutCreateWindow("Streak Line Visualization in 2D Flow behind a Cylinder.");

	// Init glew so that the GLSL functionality will be available
	if(glewInit() != GLEW_OK)
		cout << "GLEW init failed!" << endl;

	// OpenGL/GLSL initializations
	initGL();
	initGLSL();
	initArrayBuffer();
	initVertexArray();
	initFeedback();
	initFlow(flowData);
	initParams();

	glGenQueries(1, &Query);

	// Register callback functions   
	glutDisplayFunc(display);
	glutIdleFunc(idle);
	
	// Enter main loop
	glutMainLoop();

	return 0;
}
