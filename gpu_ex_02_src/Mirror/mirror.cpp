// *** Spiegelungen mit Stencil Buffer simulieren

#include <math.h>
#include <GL/freeglut.h>

GLfloat viewPos[3] = {0.0f, 2.0f, 2.0f};

#define PI 3.141592f

#define ROTATE 1
#define MOVE 2

int width = 600;
int height = 600;

float theta = PI / 2.0f - 0.01f;
float phi = 0.0f;
float distance = 2.5f;
float oldX, oldY;
int motionState;

float rotX = 0;
float rotZ = 0;

GLfloat lightPos[4] = {3, 3, 3, 1};
GLfloat mirrorColor[4] = {1.0f, 0.2f, 0.2f, 0.6f};
GLfloat teapotColor[4] = {0.8f, 0.8f, 0.2f, 1.0f};
GLfloat clearColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};

const int textureWidth = 256;
const int textureHeight = 256;

GLuint texHandle;

float compDistance(int xa,int ya,int xb,int yb)
{
	return sqrt(pow((float)xa-xb,2)+pow((float)ya-yb,2));
}


unsigned char findColor(int xa,int ya,int xb,int yb)
{
	float distance = compDistance(xa,ya,xb,yb);
	float maxDist = (float)textureWidth/2;
	float factor = distance/maxDist;
    if(factor <= 1.0f)
        return 255*(1-factor);
    else
        return 0;
}

GLuint generateTexture()
{
	int textureXCenter = (int)(textureWidth/2);
	int textureYCenter = (int)(textureHeight/2); 

    GLubyte texture[textureWidth*textureHeight*4];

	for(int i = 0;i < textureHeight;++i)
	{
		for(int j = 0;j<textureWidth;++j)
		{
			int pos = (i*textureWidth+j)*4;
			texture[pos] = 255;
            texture[pos+1] = 255;
            texture[pos+2] = 255;
            texture[pos+3] = findColor(j,i,textureXCenter,textureYCenter);
		}
	}

	GLuint textureHandle = 0;
	glGenTextures(1,&textureHandle);
	glBindTexture(GL_TEXTURE_2D,textureHandle);

	glTexImage2D(GL_TEXTURE_2D, 0, 4, textureWidth, textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, &texture[0]);
	
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);	

	glBindTexture(GL_TEXTURE_2D, 0);

	return textureHandle;
}

// Szene zeichnen: Eine Teekanne
void drawScene()
{
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, teapotColor);
	glPushMatrix();
	glTranslatef(0,0.37f,0);
	glutSolidTeapot(0.5f);
	glPopMatrix();
}

// Spiegel zeichen: Ein Viereck
void drawMirror()
{
	glPushMatrix();
		// rotate the mirror according to user input
		glRotatef(rotX, 1.0f, 0.0f, 0.0f);
		glRotatef(rotZ, 0.0f, 0.0f, 1.0f);

		glBegin(GL_QUADS);
		glVertex3f(1,0,1);
		glVertex3f(1,0,-1);
		glVertex3f(-1,0,-1);
		glVertex3f(-1,0,1);
		glEnd();
	glPopMatrix();
}

void drawTexturedMirror()
{
	glBindTexture(GL_TEXTURE_2D, texHandle);
	glPushMatrix();
		// rotate the mirror according to user input
		glRotatef(rotX, 1.0f, 0.0f, 0.0f);
		glRotatef(rotZ, 0.0f, 0.0f, 1.0f);

		glBegin(GL_QUADS);
        glNormal3f(0,1,0);
		glTexCoord2f(1,1);
		glVertex3f(1,0,1);

        glNormal3f(0,1,0);
		glTexCoord2f(0,1);		
		glVertex3f(1,0,-1);

        glNormal3f(0,1,0);
		glTexCoord2f(0,0);
		glVertex3f(-1,0,-1);

        glNormal3f(0,1,0);
		glTexCoord2f(1,0);
		glVertex3f(-1,0,1);
		glEnd();
	glPopMatrix();
	glBindTexture(GL_TEXTURE_2D, 0);
}

void display(void)	
{
	glClearColor(clearColor[0], clearColor[1], 
                 clearColor[2], clearColor[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glLoadIdentity();
	float x = distance * sin(theta) * cos(phi);
	float y = distance * cos(theta);
	float z = distance * sin(theta) * sin(phi);

	gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    // *** Spiegel zeichnen, so dass Spiegelobjekt im Stencil Buffer eingetragen wird
    // *** Framebuffer dabei auf Read-Only setzen, Depth Buffer deaktivieren, Stencil Test aktivieren
    glDisable(GL_DEPTH_TEST);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_ALWAYS, 1, 1);
    glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
    // draw mask
    drawMirror();

    // *** Gespiegelte Szene zeichnen, Stencil Buffer so einstellen, dass nur bei
    // *** einem Eintrag 1 im Stencil Buffer das entsprechende Pixel im Framebuffer
    // *** gezeichnet wird, der Inhalt vom Stencil Buffer soll unveraendert bleiben
    // *** Depth Buffer wieder anmachen, Framebuffer Maskierung deaktivieren
    // *** Was macht man mit der Lichtquelle ?
    glStencilFunc(GL_EQUAL, 1, 1);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glPushMatrix();
        // mirror along normal
        glRotatef(rotX, 1.0f, 0.0f, 0.0f);
        glRotatef(rotZ, 0.0f, 0.0f, 1.0f);
        glScalef(1.0f, -1.0f, 1.0f);

        // position light again (flip it as well)
        glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

        drawScene();
    glPopMatrix(); // *** Spiegelung der Szene rueckgaengig machen

    // *** Stencil Test deaktivieren
    // *** Spiegelobjekt mit diffuser Farbe mirrorColor zeichen
    // *** Blending aktivieren und ueber Alpha-Kanal mit Spiegelbild zusammenrechnen
    glDisable(GL_STENCIL_TEST);
	
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    /*
     * The problem with this is the way that blending works. Once something has been placed into the
     * frame buffer it can only be overwritten. That means that the full reflection is in there
     * and can't be removed to let the background show through. All we can do is place new pixels
     * on top. Maybe there is a way to do this properly. (Offscreen-rendering the scene without reflection into a texture,
     * rendering the reflection only on a white background into a texture as well,
     * doing some post-processing to separate the reflection from the background,
     * combining them and rendering them flat onto the screen?) 
     * There definitely is a way to do this using shaders,
     * but that didn't seem to be the objective of the task.
     * In any case, I don't see a way to obtain the desired effect while drawing reflection and mirror one 
     * after the other and without using shaders.
     * The current solution at least looks correct (unless we change the clear color...).
     */

    glEnable(GL_BLEND);	
    glDisable(GL_DEPTH_TEST); // no depth test, otherwise the mirror will only be drawn once
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, clearColor);
		glBlendFunc(GL_ONE_MINUS_SRC_ALPHA,GL_SRC_ALPHA);	// fades reflection (but leaves a white border...)

        glDisable(GL_LIGHTING); // no lighting, else the white border becomes visible <.<'
		drawTexturedMirror();
        glEnable(GL_LIGHTING);
        
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mirrorColor);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // fades mirror 

        drawTexturedMirror(); 
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    // Szene normal zeichnen (ohne Spiegelobjekt)
    // draw last because we had to disable the depth test while drawing the ground
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    drawScene();

	glutSwapBuffers();	
}

void keyboard(unsigned char key, int x, int y)
{
	// rotate mirror along x-axis
	if(key == 'a')
		rotX = (rotX >= 360)? 0 : rotX+1.0f;
	else if(key == 'd')
		rotX = (rotX < 0)? 359 : rotX-1.0f;
	// rotate mirror along z-axis
	else if(key == 'w')
		rotZ = (rotZ >= 360)? 0 : rotZ+1.0f;
	else if(key == 's')
		rotZ = (rotZ < 0)? 359 : rotZ-1.0f;

	glutPostRedisplay();
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
	GLfloat fogColor[] = {0.5f, 0.5f, 0.5f, 1.0f};
	GLfloat lightPos[4] = {3, 3, 3, 1};

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
	glutInitWindowSize(width, height);
	glutCreateWindow("Teapot im Spiegel");

	glutDisplayFunc(display);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutKeyboardFunc(keyboard);


	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glEnable(GL_DEPTH_TEST);

	glEnable(GL_TEXTURE_2D);

	glViewport(0,0,width,height);					
	glMatrixMode(GL_PROJECTION);					
	glLoadIdentity();								

	gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	texHandle = generateTexture();		
	//glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);	
	glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_MODULATE);

	glutMainLoop();
	return 0;
}