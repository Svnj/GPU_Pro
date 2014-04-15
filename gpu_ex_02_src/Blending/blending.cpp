#include <GL/freeglut.h>

int width = 600;
int height = 600;

int displayMode = 0;

GLfloat colors[3][4][4] = { {{0,0,0,1},{1,0,0,1},{0,1,0,1},{0,0,1,1}},
                            {{0,0,0,1},{1,0,0,0.7},{0,1,0,0.7},{0,0,1,0.7}},
                            {{1,1,1,1},{1,1,0,1},{0,1,1,1},{1,0,1,1}} };

void drawQuad(float x, float y, float z)
{
	glBegin(GL_QUADS);
	glVertex3f(x,y,z);
	glVertex3f(x+1,y,z);
	glVertex3f(x+1,y+1,z);
	glVertex3f(x,y+1,z);
	glEnd();
}


void display(void)	
{
	glClearColor(colors[displayMode][0][0], 
				 colors[displayMode][0][1], 
				 colors[displayMode][0][2], 
				 colors[displayMode][0][3]);
	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	gluLookAt(0, 0, 1, 0, 0, 0, 0, 1, 0);

	// *** Farben mit Alpha Kanal setzen
	glColor4fv(colors[displayMode][1]);
	drawQuad(1, 1, -2);
	glColor4fv(colors[displayMode][2]);
	drawQuad(0.25, 0.75, -1);
	glColor4fv(colors[displayMode][3]);
	drawQuad(0.5, 0.25, 0);

	glFlush();
}


void keyboard(unsigned char key, int x, int y)
{
	// use 1 to rotate through different display modes
	if(key == '1')
		displayMode = (displayMode + 1) % 3;

	if(displayMode == 0)
		glBlendFunc(GL_ONE, GL_ONE);
	else if(displayMode == 1)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	else
		glBlendFunc(GL_DST_COLOR, GL_ZERO);

}

void idle(void)
{
	glutPostRedisplay();
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("Blending");

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutIdleFunc(idle);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);

	glViewport(0,0,width,height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 2, 0, 2, 0, 100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// *** initial BlendFunc, rest happens in keyboard
	glBlendFunc(GL_ONE, GL_ONE);

	glutMainLoop();
	return 0;
}
