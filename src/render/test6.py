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

def draw_ground():
    size = 10
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
    glutSolidCube(1.)
    glColor4f(1.,0.,0,0.5)
    glutWireCube(1.)

    glPopMatrix()

def draw_sphere(center, radius, color = [1,0,0]):
    glPushMatrix()
    glTranslatef(*center)

    glColor3f(*color)
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

def draw(self):
    skeleton = world.skeleton
    data = skeleton.get_data()
    env_points = world.env.get_collider_points(skeleton.position, skeleton.direction)

    draw_ground()
    for box3d in env.boxes:
        draw_cube(box3d)

    draw_collider(env_points, data.environment)

    draw_skeleton(data.bone_position)

    draw_path(data.goal_position, [0,1,1])

    traj = np.zeros((len(data.trajectory_position), 3))
    traj[:,[0,2]] = data.trajectory_position
    draw_path(traj, [1,0,1])

def controls_3d(mouse_pos):
    mouse_dx, mouse_dy = pg.mouse.get_rel()

    if pg.mouse.get_pressed()[2]:
        look_speed = .1
        buffer = glGetFloatv(GL_MODELVIEW_MATRIX)
        c = (-1 * np.mat(buffer[:3, :3]) * np.mat(buffer[3, :3]).T).reshape(3, 1)

        glTranslate(*c)

        m = buffer.flatten()
        
        glRotate(mouse_dx * look_speed, m[1], m[5], m[9])
        glRotate(mouse_dy * look_speed, m[0], m[4], m[8])

        glRotatef(np.rad2deg(np.arctan2(-m[4], m[5])), m[2], m[6], m[10])

        glTranslate(*-c)

    if pg.mouse.get_pressed()[1]:
        pan_speed = .05
        buffer = glGetDoublev(GL_MODELVIEW_MATRIX)
        m = buffer.flatten()
        glTranslatef(* pan_speed * (-mouse_dy * m[[1, 5, 9]] + mouse_dx * m[[0, 4, 8]]))

    fwd,strafe = 0,0
    global running

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False            
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_q:
                running = False
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 4:
                fwd -= 0.2        
            elif event.button == 5:
                fwd += 0.2

            
    keystate = pg.key.get_pressed()
    if keystate[pg.K_w]:
        fwd -= .1
    if keystate[pg.K_s]:
        fwd += .1
    if keystate[pg.K_a]:
        strafe += .1
    if keystate[pg.K_d]:
        strafe -= .1  

    mousestate = pg.mouse.get_pressed()
    if mousestate[0]:
        world.skeleton.set_goal(mouse_pos)

    if abs(fwd) or abs(strafe):
        m = glGetDoublev(GL_MODELVIEW_MATRIX).flatten()
        glTranslate(* fwd * m[[2,6,10]])
        glTranslate(* strafe * m[[0,4,8]])

def raycast():
    m_x, m_y = pg.mouse.get_pos()
    m_y = 600 - m_y
    m_z = glReadPixels(m_x, m_y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)

    pos = gluUnProject(m_x,m_y,m_z)

    draw_sphere(pos, 0.1)

    return pos


if __name__ == "__main__":
    env = BoxEnv([
        Box(Transform.from_rot_pos(
            [10,0,0],
            [-3.5,0,2]
        ), [2,1,1]),
        Box(Transform.from_rot_pos(
            [0,20,0],
            [-3,1,2]
        ))
    ])
    obj_list = []
    world = World(env, obj_list)
    world.skeleton.set_goal([5,0,2])


    display = (800, 600)
    pg.init()
    pg.display.set_mode(display, DOUBLEBUF | OPENGL)
    clock = pg.time.Clock()

    glutInit()
    glClearColor(0.4, 0.5, 0.9, 0)
    gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_MODELVIEW)
    gluLookAt(-3,3,-3, 0,0,0, 0,1,0)


    running = True

    while running:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        

        draw(world)
        mouse_pos = raycast()
        
        controls_3d(mouse_pos)
        world.update()

        pg.display.flip()
        clock.tick(60)

    pg.quit()
