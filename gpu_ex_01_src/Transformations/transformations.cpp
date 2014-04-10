// *** Transformationen

#include <math.h>
#include <GL/freeglut.h>
#include <iostream>

#define PI 3.141592f

#define ROTATE 1
#define MOVE 2

int width = 600;
int height = 600;

float theta = PI / 2.0f - 0.4f;
float phi = 0.0f;
float distance = 40.0f;
float oldX, oldY;
int motionState;

// Winkel, der sich kontinuierlich erhöht. (Kann für die Bewegungen auf den Kreisbahnen genutzt werden)
float angle = 0.0f;

float toDeg(float angle) { return angle / PI * 180.0f; }
float toRad(float angle) { return angle * PI / 180.0f; }

// Zeichnet einen Kreis mit einem bestimmten Radius und einer gewissen Anzahl von Liniensegmenten (resolution) in der xz-Ebene.
void drawCircle(float radius, int resolution)
{
	// Abschalten der Beleuchtung.
	glDisable(GL_LIGHTING);

	// Zeichnen eines Kreises. 
	// Nutzen Sie die Methoden glBegin, glEnd, glVertex3f und ggf. glColor3f um einen GL_LINE_STRIP zu rendern.
	glBegin(GL_LINE_STRIP);
	glColor3f(0, 0, 0);
	float angle;
	for(int i = 0; i <= resolution; i++)
	{
		angle = 2*PI*i/resolution;
		glVertex3f(cos(angle)*radius, 0, sin(angle)*radius);
	}
	glEnd();

	// Anschalten der Beleuchtung.
	glEnable(GL_LIGHTING);
}

void display(void)	
{
	// Buffer clearen
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// View Matrix erstellen
	glLoadIdentity();
	float x = distance * sin(theta) * cos(phi);
	float y = distance * cos(theta);
	float z = distance * sin(theta) * sin(phi);
	gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	// Teekanne rendern.
	glutSolidTeapot(1);

	// Den Matrix-Stack sichern.
	glPushMatrix();
	
		// Zeichnen der Kugelkreisbahn.	
			drawCircle(10.0f, 50);
		// Zeichnen der Kugel.
			// Wenden Sie eine Translation und eine Rotation an, bevor sie die Kugel zeichnen. Sie können die Variable 'angle' für die Rotation verwenden.
			// Bedenken Sie dabei die richtige Reihenfolge der beiden Transformationen.
			glRotated(angle, 0, 1.0f, 0);
			glTranslatef(10.0f, 0, 0);
			glutSolidSphere(1.0f, 32, 32);
		// Zeichnen der Würfelkreisbahn.
			// Hinweis: Der Ursprung des Koordinatensystems befindet sich nun im Zentrum des Würfels.
			// Drehen Sie das Koordinatensystem um 90° entlang der Achse, die für die Verschiebung des Würfels genutzt wurde.
			// Danach steht die Würfelkreisbahn senkrecht zur Tangentialrichtung der Kugelkreisbahn.
			glRotated(90.0f, 1.0f, 0, 0);
			drawCircle(5.0f, 50);
		// Zeichnen des Würfels.
			// Wenden Sie die entsprechende Translation und Rotation an, bevor sie den Würfel zeichnen.
			glRotated(angle, 0, 1.0f, 0);
			glTranslatef(5.0f, 0, 0);
			glutSolidCube(1.0f);
		// Zeichnen einer Linie von Würfel zu Kegel.
			glDisable(GL_LIGHTING);
			glTranslatef(3.0f, 0, 0);
			glBegin(GL_LINE_STRIP);
			glVertex3f(0, 0, 0);
			glVertex3f(-3.0f, 0, 0);
			glEnd();
			glEnable(GL_LIGHTING);
		// Drehung anwenden, sodass Koordinatensystem in Richtung Ursprung orientiert ist. (Hinweis: Implementieren Sie dies zuletzt.)
			GLfloat height = 8*cos(toRad(angle-90));
			GLfloat d = 8*sin(toRad(angle-90));
			GLfloat e = 10 - d;
			GLfloat l = pow(pow(height,2) + pow(e,2), 0.5f);
			GLfloat alpha = toDeg(acos(e/l)) - 90;
			if(static_cast<int>(angle)%360 > 180) alpha = -alpha;
			glRotated(alpha, 0, 1.0f, 0);
			if(static_cast<int>(angle)%360 > 180)
			{
				glRotated(180 - angle, 0, 1.0f, 0);
				std::cout << alpha + 180 - (int)angle%360 << std::endl;
			}
			else 
			{
				glRotated(-angle, 0, 1.0f, 0);
				std::cout << alpha - (int)angle%360 << std::endl;
			}
		// Zeichnen der Linie von Kegel zu Urpsrung.	
			glDisable(GL_LIGHTING);
			glBegin(GL_LINE_STRIP);
			glVertex3f(0, 0, 0);
			glVertex3f(0, 0, l);
			glEnd();
			glEnable(GL_LIGHTING);
		// Zeichnen des Kegels.
			glutSolidCone(0.5f, 1.0f, 32, 4);
	// Den Matrix-Stack wiederherstellen.

	glPopMatrix();
	
	glutSwapBuffers();	

	angle += 5.0f / 60.0f;
}

void mouseMotion(int x, int y)
{
	float deltaX = x - oldX;
	float deltaY = y - oldY;
	
	if (motionState == ROTATE) {
		theta -= 0.01f * deltaY;

		if (theta < 0.01f) theta = 0.01f;
		else if (theta > PI/2.0f - 0.01f) theta = PI/2.0f - 0.01f;

		phi += 0.01f * deltaX;	
		if (phi < 0) phi += 2*PI;
		else if (phi > 2*PI) phi -= 2*PI;
	}
	else if (motionState == MOVE) {
		distance += 0.01f * deltaY;
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

void idle(void)
{
	glutPostRedisplay();
}


int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("Transformationen");

	glutDisplayFunc(display);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutIdleFunc(idle);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	
	glEnable(GL_DEPTH_TEST);

	glViewport(0,0,width,height);					
	glMatrixMode(GL_PROJECTION);					
	glLoadIdentity();								

	gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glutMainLoop();
	return 0;
}
