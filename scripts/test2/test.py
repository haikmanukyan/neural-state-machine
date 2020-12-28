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
        glutCreateWindow("Hello")
        gluPerspective(45, (640 / 480), 0.1, 1000)
        glClearColor(0.5,0.7,0.9,1)
        glPushMatrix()

        glutKeyboardFunc(self.onKeyDown)
        glutMotionFunc(self.onDrag)

        self.camera = np.array([-2,-2,-2])
        self.center = [0,1,0]
        self.up = [0,1,0]
        gluLookAt(*self.camera,*self.center,*self.up)

        self.frameRate = 60
        self.draw = draw_fn
        self.update = update_fn

    def onDrag(self, x, y):
        # print (x / self.size[0] - 0.5)
        # self.camera[0] += x / self.size[0] - 0.5
        # self.camera[2] -= x / self.size[0] - 0.5
        # self.camera[1] += y / self.size[1] - 0.5
        # self.camera = 2 * self.camera / la.norm(self.camera)
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

        glPopMatrix()
        glPushMatrix()
        gluLookAt(*self.camera,*self.center,*self.up)

        if self.draw is not None:
            self.draw()
        glutSwapBuffers()

    def run(self):
        glutDisplayFunc(self.display)
        glutIdleFunc(self.step)
        glutMainLoop()