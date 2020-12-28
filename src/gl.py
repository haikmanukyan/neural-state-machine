import sys
sys.path.append('.')

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

from src.env.World import World
from src.env.Environment import Box, BoxEnv
from src.env.Transform import Transform
from scipy.spatial.transform import Rotation

import math
import pygame as pg
from pygame import key
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def draw_ground(size = 10):
    glBegin(GL_QUADS)
    glColor3fv([0,1,0])
    glVertex(-size,0,-size)
    glVertex(size,0,-size)
    glVertex(size,0,size)
    glVertex(-size,0,size)
    glEnd()

def draw_cube(box):
    glPushMatrix()
    
    glTranslatef(*box.transform.position())
    glTranslatef(* (np.array(box.size) * 0.5))

    rotvec = Rotation.from_matrix(box.transform.rotation()).as_rotvec()
    glRotatef(np.rad2deg(np.linalg.norm(rotvec)), *rotvec)

    glScalef(*box.size)

    glColor4f(1.,1.,0,0.5)
    glutSolidCube(1.0)
    glColor4f(1.,0.,0,0.5)
    glutWireCube(1.)

    glPopMatrix()

def draw_sphere(center, radius, color = [1,0,0]):
    glPushMatrix()
    glTranslatef(*center)

    glColor3f(*color)
    # quadric = gluNewQuadric()
    # gluSphere(quadric, radius, 6, 6)
    glutWireSphere(radius, 6, 4)
    glPopMatrix()

def draw_collider(points, radius = 0.5):
    if not hasattr(radius, "__len__"):
        radius = [radius] * len(points)
    for point, r in zip(points, radius):
        if r == 0: continue
        draw_sphere(point, r * 0.1)

def draw_path(points, color = [1,0,0]):
    glLineWidth(8.0)
    glColor3f(*color)

    glBegin(GL_LINES)
    
    for i in range(len(points) - 1):
        glVertex3fv(points[i])
        glVertex3fv(points[i+1])

    glEnd()

    for point in points:
        draw_sphere(point, 0.05, color)

    glLineWidth(1.0)
    

def draw_skeleton(pose, color = [1,1,1]):
    bone_order = list(range(7)) + [5] + list(range(7,11)) \
        + list(range(10,6,-1)) + [5] + list(range(11,15)) \
        + list(range(14,10,-1)) + list(range(5,-1,-1)) \
            + list(range(15,19)) + list(range(18,14,-1)) + [0] \
                + list(range(19,23))

    glLineWidth(3.0)
    glColor3f(*color)

    glBegin(GL_LINES)
    
    glVertex3fv(pose[bone_order[0]])
    for v in pose[bone_order]:
        glVertex3fv(v)
        glVertex3fv(v)
    glVertex3fv(pose[bone_order[-1]])

    glEnd()

    glLineWidth(1.0)

def draw_world(world):
    skeleton = world.skeleton
    data = skeleton.get_data()
    env_points = world.env.get_collider_points(skeleton.position, skeleton.direction)

    draw_ground()
    for box in world.env.boxes:
        draw_cube(box)
    mouse_pos = raycast()

    draw_collider(env_points, data.environment)
    draw_skeleton(data.bone_position)
    draw_path(data.goal_position, [0,1,1])

    traj = np.zeros((len(data.trajectory_position), 3))
    traj[:,[0,2]] = data.trajectory_position
    draw_path(traj, [1,0,1])
    
    return mouse_pos

def raycast():
    m_x, m_y = pg.mouse.get_pos()
    m_y = 600 - m_y
    m_z = glReadPixels(m_x, m_y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)

    pos = gluUnProject(m_x,m_y,m_z)
    draw_sphere(pos, 0.1, (0,0,1))

    return pos

if __name__ == "__main__":
    from OpenGL.GLUT import *
    from time import sleep

    
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutCreateWindow("Window")

    quadric=gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluQuadricTexture(quadric, GL_TRUE)
    
    animationAngle = 0.0
    frameRate = 25

    def doAnimationStep( ):
        global animationAngle
        global frameRate
        animationAngle += 1
        while animationAngle > 360:
            animationAngle -= 360
        sleep( 1 / float( frameRate ) )
        glutPostRedisplay( )

    def display():
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glMatrixMode( GL_MODELVIEW )
        glLoadIdentity( )
        glColor3f( 1, 1, 1 )
        glRotatef( animationAngle, 0, 0, 1 )
        glBegin( GL_QUADS )
        glVertex3f( -0.5, 0.5, 0 )
        glVertex3f( -0.5, -0.5, 0 )
        glVertex3f( 0.5, -0.5, 0 )
        glVertex3f( 0.5, 0.5, 0 )
        glEnd(  )
        glutSwapBuffers( )

    glutDisplayFunc(display)
    glutIdleFunc( doAnimationStep )
    glutMainLoop()
