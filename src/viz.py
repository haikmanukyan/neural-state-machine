import sys
sys.path.append('.')

from src.env.Skeleton import initial_pose
from src.gl import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

import time
import numpy as np
import numpy.linalg as la

class GLWindow:
    def __init__(self, draw_fn=None, update_fn=None):
        self.size = (640,480)

        glutInit()
        glutInitWindowSize(640,480)

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutCreateWindow("Model")
        glEnable(GL_DEPTH_TEST)
        gluPerspective(45, (640 / 480), 0.1, 1000)
        glClearColor(0.5,0.7,0.9,1)
        glPushMatrix()

        glutKeyboardFunc(self.onKeyDown)
        glutMotionFunc(self.onDrag)
        glutMouseFunc(self.onMouse)

        self.camera = np.array([-2.2,2.2,2.2])
        self.center = [0,1,0]
        self.up = [0,1,0]
        gluLookAt(*self.camera,*self.center,*self.up)

        self.frameRate = 60
        self.draw = draw_fn
        self.update = update_fn

        self.x, self.y = 0,0
        self.h = 2
        self.zoom = 4
        self.frame_idx = 0

    def onMouse(self, button, state, x, y):
        x, y = x / self.size[0] - 0.5, y / self.size[1] - 0.5
        self.x, self.y = x,y

        if button == 3:
            self.zoom -= 0.5
        if button == 4:
            self.zoom += 0.5
        self.zoom = np.clip(self.zoom, 0.1, 100.)
        self.camera = self.camera / la.norm(self.camera) * self.zoom

    def onDrag(self, x, y):
        x, y = x / self.size[0] - 0.5, y / self.size[1] - 0.5
        x_sens, y_sens = 10.,100.
        dx, dy = x - self.x, y - self.y
        self.x, self.y = x, y


        polar = np.arctan2(self.camera[2], self.camera[0])
        polar += x_sens * dx
        self.camera[0] = 2 * np.cos(polar)
        self.camera[2] = 2 * np.sin(polar)

        self.h += y_sens * dy
        self.camera[1] = self.h 

        self.camera = self.camera / la.norm(self.camera) * self.zoom
        
        pass

    def onKeyDown(self, key, x, y):
        glutLeaveMainLoop()

    def step(self):
        time.sleep(1 / float(self.frameRate))
        if self.update is not None:
            self.update()
        glutPostRedisplay()

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glColor3f(0,0,0)
        glLoadIdentity()
        gluOrtho2D(0,640,0,480)
        glRasterPos2d(10,450)
        glutBitmapString(GLUT_BITMAP_HELVETICA_18, b"%d" % self.frame_idx)

        glPopMatrix()
        glPushMatrix()
        gluLookAt(*self.camera,*self.center,*self.up)

        draw_ground(2)
        if self.draw is not None:
            self.draw()

        glutSwapBuffers()

    def run(self):
        glutDisplayFunc(self.display)
        glutIdleFunc(self.step)
        glutMainLoop()