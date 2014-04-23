// VBO_Fractal
// TODO: Check if GLEW is required, remove ARB extension

#include <iostream>
#include <vector>
#include <cstdio>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Vector.h"
#include "Triangle.h"


using namespace std;

float center[3] = {0.0f, 0.5f, 0.0f}; 

#define PI 3.141592f

#define ROTATE 1
#define MOVE 2

float thetaStart = PI / 2.0f - 0.5f;
float phiStart = PI / 2.0f;
float rStart = 40.0f;

float theta = thetaStart;
float phi = phiStart;
float r = rStart;

float oldX, oldY;
int motionState;

bool useVBO = false;

// Window size

int width = 800;       
int height = 800;

// view parameters

GLfloat viewPosition[4] = {0.0f, 5.0f, 8.0f, 1.0f};  
GLfloat viewDirection[4] = {-0.0f, -5.0f, -8.0f, 0.0f};  
GLfloat viewAngle = 40.0f;
GLfloat viewNear = 0.2f;
GLfloat viewFar = 100000.0f;

GLfloat lightPosition[4] = {0, 0, 1 ,0};  

vector<int> indices;
vector<Vector> positions;
vector<Vector> normals;
vector<Triangle> triangles;

GLuint positionBufferHandle;
GLuint normalBufferHandle;
GLuint indexBufferHandle;
GLuint vertexArrayHandle;

GLuint displayListHandle = 1;
GLuint rendermode = 0;

Triangle startTriangle(Vector(0, 0, 0), Vector(0, 0, 1), Vector(1, 0, 0));

Vector startP0(-0.5f, 0.0f, 0.5f*sqrt(3.0f));
Vector startP1(1.0f, 0.0f, 0.0f);
Vector startP2(-0.5f, 0.0f, -0.5f*sqrt(3.0f));
Vector startP3(0.0f, 1.0f, 0.0f);

int globalIndex = 0;
int maxLevel = 8;

unsigned int positionsVBO;
unsigned int normalsVBO;
unsigned int indicesVBO;

unsigned int VAO;

void generateTriangle(Vector p0, Vector p1, Vector p2)
{
	Triangle tri = Triangle(p0, p1, p2); 
	positions.push_back(p0);
	positions.push_back(p1);
	positions.push_back(p2);

	normals.push_back(tri.getNormal());
	normals.push_back(tri.getNormal());
	normals.push_back(tri.getNormal());

	indices.push_back(globalIndex);
	globalIndex++;
	indices.push_back(globalIndex);
	globalIndex++;
	indices.push_back(globalIndex);
	globalIndex++;
}

void generateTetraeders(Vector p0, Vector p1, Vector p2, Vector p3, int level)
{	
	if (level == maxLevel)
		return;

	generateTriangle(p0, p1, p3);
	generateTriangle(p1, p2, p3);
	generateTriangle(p2, p0, p3);
	generateTriangle(p0, p2, p1);

	Vector e0 = p1 - p0;
	Vector e1 = p2 - p1;
	Vector e2 = p2 - p0;
	Vector e3 = p3 - p0;
	Vector e4 = p3 - p2;
	Vector e5 = p3 - p1;
	
	Vector newP0, newP1, newP2, newP3;

	newP0 = p0;
	newP1 = p0 - 0.5 * e0;
	newP2 = p0 - 0.5 * e2;
	newP3 = p0 - 0.5 * e3;

	generateTetraeders(newP0, newP2, newP1, newP3, level+1);

	newP0 = p1 + 0.5 * e0;
	newP1 = p1;
	newP2 = p1 - 0.5 * e1;
	newP3 = p1 - 0.5 * e5;

	generateTetraeders(newP1, newP0, newP2, newP3, level+1);

	newP0 = p2 + 0.5 * e2;
	newP1 = p2 + 0.5 * e1;
	newP2 = p2;
	newP3 = p2 - 0.5 * e4;

	generateTetraeders(newP2, newP1, newP0, newP3, level+1);

	newP0 = p3 + 0.5 * e3;
	newP1 = p3 + 0.5 * e5;
	newP2 = p3 + 0.5 * e4;
	newP3 = p3;

	generateTetraeders(newP0, newP2, newP1, newP3, level+1);
}

void generateGeometryVertexBuffer()
{
	// TODO: Erzeugen eines Vertex Buffer Objects für die Positionen.
	// Hinweis: Die Erzeugung eiens VBOs vollzieht sich in 4 Schritten
	// - Generieren eines Buffer Handles
	// - Binden des Buffers (Target = GL_ARRAY_BUFFER)
	// - Buffer Daten in den VRAM kopieren (Usage = GL_STATIC_DRAW)
    // - Buffer nicht mehr binden (stattdessen eine 0 binden).
    unsigned int positionsVBOSizeInByte = sizeof(positions[0])*positions.size();
	glGenBuffers(1,&positionsVBO);	
	glBindBuffer(GL_ARRAY_BUFFER,positionsVBO);
    glBufferData(GL_ARRAY_BUFFER,positionsVBOSizeInByte,&positions[0],GL_STATIC_DRAW);
	glVertexPointer(3,GL_FLOAT,0,NULL);
	glBindBuffer(GL_ARRAY_BUFFER,0);

	// TODO: Erzeugen eines Vertex Buffer Objects für die Normalen.
    unsigned int normalsVBOSizeInByte = sizeof(normals[0])*normals.size();
	glGenBuffers(1,&normalsVBO);	
	glBindBuffer(GL_ARRAY_BUFFER,normalsVBO);
    glBufferData(GL_ARRAY_BUFFER,normalsVBOSizeInByte,&normals[0],GL_STATIC_DRAW);
	glNormalPointer(GL_FLOAT,0,NULL);
	glBindBuffer(GL_ARRAY_BUFFER,0);
	
	// TODO: Erzeugen eines Indexbuffers. Dies funktioniert genauso, wie die Erzeugung eines VBOs (Hinweis: Target = GL_ELEMENT_ARRAY_BUFFER).
    unsigned int indicesVBOSizeInByte = sizeof(indices[0])*indices.size();
	glGenBuffers(1,&indicesVBO);	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,indicesVBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,indicesVBOSizeInByte,&indices[0],GL_STATIC_DRAW);
	glIndexPointer(GL_UNSIGNED_INT,0,NULL);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);	
	
	
	// TODO: Erzeugen eines Vertex Array Objects (VAOs).
	// VAOs werden genutzt um der GPU mitzuteilen, von welchen VBOs die Daten für das Rendering genommen werden sollen.
	// - Generieren eines Vertex Array Handles
	// - Binden des Vertex-Array Handles
		// Binden des VBOs, von dem die Positionen geholt werden sollen
		// mit glVertexAttribPointer der GPU mitteilen, wie das VBO interpretiert werden soll. Im folgenden werden die Parameter kurz erklärt.
		//   Der erste Parameter ist der Index des Vertex-Attributes. (0=Position, 2=Normale)
		//   Daraufhin folgen die Dimension der Daten (z.B. bei Positions = 3 (wegen xyz)) und der Datentyp.
		//   Die Eingabedaten müssen nicht normalisiert werden.
		//   Der Stride misst den Offset in Byte zwischen zwei Datenwerten im Buffer.
		//   Der letzte Wert ist der BufferOffset, der angewendet werden soll. In unserem Beispiel ist er immer 0, da wir für jedes Attribut einen eigenes VBO nutzen.
		// Binden des VBOs, von dem die Positionen geholt werden sollen
		// wieder glVertexAttribPointer nutzen..
		// Aufräumen: Den Wert '0' als Vertex Buffer Object binden. (kein VBO mehr aktiv)
		// Zuletzt muss mitgeteilt werden, welche Vertex-Attribut Indices genutzt werden sollen. Dies geschieht mit glEnableVertexAttribArray.
	// Aufräumen: Unbinden des Vertex Array Objects (eine 0 binden).

	glGenVertexArrays(1,&VAO);
	glBindVertexArray(VAO);
	
	glBindBuffer(GL_ARRAY_BUFFER,positionsVBO);
    glVertexAttribPointer(0,3,GL_FLOAT,false,0,NULL);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER,0);

	glBindBuffer(GL_ARRAY_BUFFER,normalsVBO);
    glVertexAttribPointer(3,3,GL_FLOAT,false,0,NULL);
	glEnableVertexAttribArray(3);
	glBindBuffer(GL_ARRAY_BUFFER,0);

	/*
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,indicesVBO);
	glVertexAttribPointer(2,1,GL_UNSIGNED_INT,false,0,NULL);
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
	*/

	glBindVertexArray(0);
}

void generateDisplayList(GLuint listNumbr)
{
	glNewList(listNumbr, GL_COMPILE);
		glBegin(GL_TRIANGLES);
		for (unsigned int i = 0 ; i < positions.size() ; i += 3) {
			glNormal3fv(&normals[i][0]);
			glVertex3fv(&positions[i][0]);
			glNormal3fv(&normals[i+1][0]);
			glVertex3fv(&positions[i+1][0]);
			glNormal3fv(&normals[i+2][0]);
			glVertex3fv(&positions[i+2][0]);
		}
		glEnd();
	glEndList();
}


void drawGeometryVertexBuffer()
{
	// TODO: Binden des VAOs. (Die GPU weiß nun, wo sie die Geometriedaten herzuholen hat.)
	glBindVertexArray(VAO);
	// TODO: Index-Buffer binden.
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,indicesVBO);
	
	// TODO: mit glDrawElements die Triangles rendern. 
	// Hinweis: Der Type-Parameter definiert den im Index Buffer verwendeten Datentyp.
	//   Der letzte Parameter dieses DrawCalls ist NULL, da der Index-Buffer von Beginn an gelesen werden soll.
	unsigned int numTriangles = positions.size();

	glDrawElements(GL_TRIANGLES,numTriangles,GL_UNSIGNED_INT,NULL);
	
	// TODO: Unbinden von VAO und Index-Buffer	
	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
}

// calc the view position and direction from theta/phi coordinates
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
}

void display()
{
	int timeStart = glutGet(GLUT_ELAPSED_TIME);

	// clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	calcViewerCamera(theta, phi, r);

	// now render from camera view
	glLoadIdentity();
 	glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
  				
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(viewPosition[0], viewPosition[1], viewPosition[2],
			  viewPosition[0] + viewDirection[0], viewPosition[1] + viewDirection[1], viewPosition[2] + viewDirection[2], 
			  0, 1, 0);


	
	switch (rendermode) 
	{
		case 0:
			// fast drawing
			drawGeometryVertexBuffer();
			break;
		case 1:
			// simple and slow drawing		
			glBegin(GL_TRIANGLES);
			for (unsigned int i = 0 ; i < positions.size() ; i += 3) {
				glNormal3fv(&normals[i][0]);
				glVertex3fv(&positions[i][0]);
				glNormal3fv(&normals[i+1][0]);
				glVertex3fv(&positions[i+1][0]);
				glNormal3fv(&normals[i+2][0]);
				glVertex3fv(&positions[i+2][0]);
			}
			glEnd();
			break;
		case 2:
			// DisplayLists
			if(!glIsList(1))
			{
				glNewList(1, GL_COMPILE);
				glBegin(GL_TRIANGLES);
				for (unsigned int i = 0 ; i < positions.size() ; i += 3) {
					glNormal3fv(&normals[i][0]);
					glVertex3fv(&positions[i][0]);
					glNormal3fv(&normals[i+1][0]);
					glVertex3fv(&positions[i+1][0]);
					glNormal3fv(&normals[i+2][0]);
					glVertex3fv(&positions[i+2][0]);
				}
				glEnd();
				glEndList();
			}
			else
			{
				glCallList(1);
			}
			break;
	}

	// glutSolidTeapot(1.0);

	// swap display buffers
	glutSwapBuffers();
	glFinish();

	// measure frame time in milliseconds
	int timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("rendermode: %d  ,Delay %d     \r",rendermode,timeEnd - timeStart);
	//printf("Delay %d     \r",timeEnd - timeStart);
}

// use a virtual trackball as mouse control
void mouseMotion(int x, int y)
{
	float deltaX = x - oldX;
	float deltaY = y - oldY;
	
	if (motionState == ROTATE) {
		theta -= 0.002f * deltaY;

		if (theta < 0.002f) theta = 0.002f;
		else if (theta > PI - 0.002f) theta = PI - 0.002f;

		phi += 0.002f * deltaX;	
		if (phi < 0) phi += 2*PI;
		else if (phi > 2*PI) phi -= 2*PI;
	}
	else if (motionState == MOVE) {
		r += 0.05f * deltaY;
		if (r < 0.01f) r = 0.01f;
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
	if(key == ' ')
	{
		rendermode = (rendermode + 1)%3;
	}	
}
void initGL()
{
	// init some GL state variables
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_LIGHTING);               
	glEnable(GL_LIGHT0);                 

	glViewport(0,0,width,height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    gluPerspective(viewAngle, 1.0f, viewNear, viewFar);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClearColor(0.5, 0.5, 1.0, 1.0);
}


int main(int argc, char** argv)
{
	// Initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(width, height);

	glutCreateWindow("Fractal with VBO");

	// Init glew so that the GLSL functionality will be available
	if(glewInit() != GLEW_OK)
	   cout << "GLEW init failed!" << endl;

	// Register GLUT callback functions   
	glutDisplayFunc(display);
	glutIdleFunc(display);
	// glutSpecialFunc(special);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);

	generateDisplayList(displayListHandle);

	// subdivideTriangle(startTriangle, 4.0, 0);
	generateTetraeders(5.0*startP0, 5.0*startP1, 5.0*startP2, 5.0*startP3, 0);
	
	// we draw the geometry as a vertex buffer
	generateGeometryVertexBuffer();
	
	initGL();
	// Enter main loop
	glutMainLoop();

	return 0;



}

