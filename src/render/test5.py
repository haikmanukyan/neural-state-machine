import pygame as pg
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

cubeVertices = ((1,1,1),(1,1,-1),(1,-1,-1),(1,-1,1),(-1,1,1),(-1,-1,-1),(-1,-1,1),(-1,1,-1))
cubeEdges = ((0,1),(0,3),(0,4),(1,2),(1,7),(2,5),(2,3),(3,6),(4,6),(4,7),(5,6),(5,7))
cubeQuads = ((0,3,6,4),(2,5,6,3),(1,2,5,7),(1,0,4,7),(7,4,6,5),(2,3,0,1))
cubeNormals = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
colors = ((1,0,0),(0,1,0),(0,0,1),(1,0,1),(1,1,0),(0,1,1))

def wireCube():
    glBegin(GL_LINES)
    for cubeEdge in cubeEdges:
        for cubeVertex in cubeEdge:
            glVertex3fv(cubeVertices[cubeVertex])
    glEnd()

def solidCube():
    glBegin(GL_QUADS)
    for cubeQuad,color,normal in zip(cubeQuads,colors,cubeNormals):
        glColor3fv(colors[0])
        for cubeVertex in cubeQuad:
            glVertex3fv(cubeVertices[cubeVertex])
            glNormal3fv(normal)
    glEnd()

def main():
    pg.init()
    display = (1680, 1050)
    pg.display.set_mode(display, DOUBLEBUF | OPENGL)

    glClearColor(0.1, 0.1, 0.1, 0)
    glShadeModel(GL_SMOOTH)
    
    # glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0,1,1,1])
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [1.0,1,1,1])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, [50.])

    glLightfv(GL_LIGHT0, GL_POSITION, [2.,2,2,0])
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_NORMALIZE)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    glRotatef(0, 0, 0, 1)

    glutInit()
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE | GLUT_DEPTH)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
            if event.type == pg.KEYDOWN:
                pg.quit()
                quit()

        glRotatef(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # glColor3fv([1.,1.,0])
        # glutSolidCube(1.)
        # glColor3fv([1.,0.,0])
        # glutSolidCube(1.)
        
        # glColor3fv([1.,1.,0])
        # glutSolidSphere(0.5,12,12)
        # glColor3fv([1.,0.,0])
        # glutWireSphere(0.5,12,12)

        # glColor3fv([0,1,0])
        # glutSolidCube(1)
        # glColor3fv([1,0,0])
        # glutWireCube(1)
        
        # glPushMatrix()
        
        # glTranslatef(1,0,0,0)
        # glutWireCube(1)
        
        # glPopMatrix()
        solidCube()

        pg.display.flip()
        pg.time.wait(10)

if __name__ == "__main__":
    main()