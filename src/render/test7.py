from __future__ import division
import pygame
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *


class Spectator:
    def __init__(self, w=640, h=480, fov=75):
        self.init()
        pygame.init()
        pygame.display.set_mode((640, 480), pygame.OPENGL | pygame.DOUBLEBUF)
        glMatrixMode(GL_PROJECTION)
        aspect = w / h
        gluPerspective(fov, aspect, 0.001, 100000.0)
        glMatrixMode(GL_MODELVIEW)

    def init(self):
        self.points = np.array([2.4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 2.4, 0, 0,
                                    2.4, 0, 2, 0, 0, 2, 0, -1.66, 0, 0, 0, 0, 0, -1.66, 0, 2.4, 0, 0, 0, 0, 0,
                                    2.4, 0, 0, 2.4, -1.66, 2, 2.4, 0, 2, 2.4, -1.66, 2, 0, 0, 2, 2.4, 0, 2, 0,
                                    -1.66, 0, 0, 0, 2, 0, -1.66, 2, 2.4, 0, 0, 0, -1.66, 0, 2.4, -1.66, 0,
                                    2.4, -1.66, 2, 2.4, 0, 0, 2.4, -1.66, 0, 0, 0, 2, 2.4, -1.66, 2, 0,
                                    -1.66, 2, 2.4, -1.66, 2, 0, -1.66, 0, 0, -1.66, 2, 0, -1.66, 0, 2.4,
                                    -1.66, 2, 2.4, -1.66, 0], 'f').reshape(-1, 3)

        self.normals = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                                    1, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 1, 0, 0, 1,
                                    0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0,
                                    -1, 0, 0, -1, 0, 0, -1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                                    0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0
                                    ], 'f').reshape(-1, 3)

        self.indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                    32, 33, 34, 35], 'i')


    def simple_lights(self):
        glEnable(GL_LIGHTING)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.9, 0.45, 0.0, 1.0))
        glLightfv(GL_LIGHT0, GL_POSITION, (0.0, 10.0, 10.0, 10.0))
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

    def simple_camera_pose(self):
        """ Pre-position the camera (optional) """
        glMatrixMode(GL_MODELVIEW)
        gluLookAt(-3,3,-3, 0,0,0,0, 1, 0)

        # glLoadMatrixf(np.array([0.741, -0.365, 0.563, 0, 0, 0.839, 0.544,
                                #    0, -0.671, -0.403, 0.622, 0, -0.649, 1.72, -4.05, 1]))

    def draw_simple_cube(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointerf(self.points)
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointerf(self.normals)
        glDrawElementsui(GL_TRIANGLES, self.indices)

    def loop(self):
        pygame.display.flip()
        pygame.event.pump()
        self.keys = dict((chr(i), int(v)) for i, v in enumerate(pygame.key.get_pressed()) if i < 256)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        return True

    def controls_3d(self, mouse_button=1, w_key='w', s_key='s', a_key='a', d_key='d'):
        """ The actual camera setting cycle """
        mouse_dx, mouse_dy = pygame.mouse.get_rel()

        if pygame.mouse.get_pressed()[mouse_button]:
            look_speed = .2
            buffer = glGetDoublev(GL_MODELVIEW_MATRIX)
            c = (-1 * np.mat(buffer[:3, :3]) *
                 np.mat(buffer[3, :3]).T).reshape(3, 1)
            glTranslate(*c)
            m = buffer.flatten()
            glRotate(mouse_dx * look_speed, m[1], m[5], m[9])
            glRotate(mouse_dy * look_speed, m[0], m[4], m[8])

            glRotated(-math.atan2(-m[4], m[5]) * 180 / np.pi, m[2], m[6], m[10])
            glTranslate(*-c)

        if pygame.mouse.get_pressed()[1]:
            pan_speed = .05
            buffer = glGetDoublev(GL_MODELVIEW_MATRIX)
            m = buffer.flatten()
            glTranslatef(* pan_speed * (-mouse_dy *
                                        m[[1, 5, 9]] + mouse_dx * m[[0, 4, 8]]))

        fwd = .1 * (self.keys[w_key] - self.keys[s_key])
        strafe = .1 * (self.keys[a_key] - self.keys[d_key])
        if abs(fwd) or abs(strafe):
            m = glGetDoublev(GL_MODELVIEW_MATRIX).flatten()
            glTranslate(fwd * m[2], fwd * m[6], fwd * m[10])
            glTranslate(strafe*m[0], strafe * m[4], strafe * m[8])


fps = Spectator(w=640, h=480, fov=75)
fps.simple_lights()
fps.simple_camera_pose()

while fps.loop():
    fps.draw_simple_cube()
    fps.controls_3d(2, 'w', 's', 'a', 'd')
    if fps.keys['q']:
        break
