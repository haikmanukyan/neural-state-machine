import sys
sys.path.append('.')

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

from src.env.Environment import Box, BoxEnv
from src.env.Transform import Transform
from src.env.World import World
from src.env.Skeleton import Skeleton
from src.nn.controller import NSMController, Seq2SeqController
from src.gl import *

from scripts.train_seq2seq import EncoderRNN, DecoderRNN


import math
import pygame as pg
from pygame import key
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class OpenGLRenderer:
    def __init__(self, world):
        self.running = False
        self.world = world

    def control(self, mouse_pos):
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
            action = 1
            if pg.key.get_mods() & pg.KMOD_CTRL:
                action = 5
            if pg.key.get_mods() & pg.KMOD_SHIFT:
                action = 2

            world.skeleton.set_goal(mouse_pos, action)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_q or event.key == pg.K_ESCAPE:
                    self.running = False
            if event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 4:
                    fwd -= 0.2        
                elif event.button == 5:
                    fwd += 0.2        

        if abs(fwd) or abs(strafe):
            m = glGetDoublev(GL_MODELVIEW_MATRIX).flatten()
            glTranslate(* fwd * m[[2,6,10]])
            glTranslate(* strafe * m[[0,4,8]])


    def gl_init(self):
        self.running = True
        self.display = (800, 600)
        self.clock = pg.time.Clock()
        
        pg.init()
        pg.display.set_mode(self.display, DOUBLEBUF | OPENGL)

        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)

        glClearColor(0.4, 0.5, 0.9, 0)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 100.0)
        glEnable(GL_DEPTH_TEST)
        # glEnable(GL_LIGHTING)

        glMatrixMode(GL_MODELVIEW)
        gluLookAt(-3,3,-3, 0,0,0, 0,1,0)

    def loop(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
        mouse_pos = draw_world(self.world)
        
        self.control(mouse_pos)
        self.world.update()

        pg.display.flip()
        self.clock.tick(60)


    def run(self):
        self.gl_init()
        while self.running:
            self.loop()
        pg.quit()


if __name__ == "__main__":
    env = BoxEnv([
        Box(Transform.from_rot_pos(
            [10,0,0],
            [-3.5,0,2]
        ), [2,1,1]),
        Box(Transform.from_rot_pos(
            [0,20,0],
            [-3,1,2]
        )),
        Box(Transform.from_rot_pos(
            [0,0,0],
            [5,0,0]
        ), 
        (0.5,0.5,0.5)),
        Box(Transform.from_rot_pos(
            [0,0,0],
            [5.5,0,0]
        ), 
        (0.1,1.0,0.5))
    ])
    obj_list = []
    world = World(env, obj_list)

    # world.skeleton = Skeleton([0,0],[0,1],Seq2SeqController())
    world.skeleton = Skeleton([0,0],[0,1])
    world.skeleton.set_goal([1,0,0],0)
    
    renderer = OpenGLRenderer(world)

    renderer.run()