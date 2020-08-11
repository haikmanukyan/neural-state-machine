from pyglet.gl import *
from pyglet.window import key
from pyglet.window import mouse
import math

class Box:
    def __init__(self, box):
        pass





class Block:
    def get_tex(self, file):
        tex = pyglet.image.load(file).get_texture()
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        return pyglet.graphics.TextureGroup(tex)

    def __init__(self):
        self.top = self.get_tex('src/render/grass_top.png')
        self.side = self.get_tex('src/render/grass_side.png')
        self.bottom = self.get_tex('src/render/dirt.png')

        self.batch = pyglet.graphics.Batch()

        tex_coords = ('t2f', (0, 0, 1, 0, 1, 1, 0, 1, ))

        x, y, z = 0, 0, -1
        X, Y, Z = x+1, y+1, z+1

        self.batch.add(4, GL_QUADS, self.top, ('v3f', (x, y, z, x, y, Z, x, Y, Z, x, Y, z, )), tex_coords)
        self.batch.add(4, GL_QUADS, self.top, ('v3f', (X, y, Z, X, y, z, X, Y, z, X, Y, Z, )), tex_coords)
        self.batch.add(4, GL_QUADS, self.top, ('v3f', (x, y, z, X, y, z, X, y, Z, x, y, Z, )), tex_coords)
        self.batch.add(4, GL_QUADS, self.top, ('v3f', (x, Y, Z, X, Y, Z, X, Y, z, x, Y, z, )), tex_coords)
        self.batch.add(4, GL_QUADS, self.top, ('v3f', (X, y, z, x, y, z, x, Y, z, X, Y, z, )), tex_coords)
        self.batch.add(4, GL_QUADS, self.top, ('v3f', (x, y, Z, X, y, Z, X, Y, Z, x, Y, Z, )), tex_coords)

    def draw(self):
        self.batch.draw()


class Player:
    def __init__(self, pos=(0, 0, 0), rot=(0, 0)):
        self.pos = list(pos)
        self.rot = list(rot)

    def mouse_motion(self, dx, dy):
        dx /= 8
        dy /= 8
        self.rot[0] += dy
        self.rot[1] -= dx
        if self.rot[0] > 90:
            self.rot[0] = 90
        elif self.rot[0] < -90:
            self.rot[0] = -90

    def update(self, dt, keys):
        s = dt*10
        rotY = -self.rot[1]/180*math.pi
        dx, dz = s*math.sin(rotY), s*math.cos(rotY)
        if keys[key.W]:
            self.pos[0] += dx
            self.pos[2] -= dz
        if keys[key.S]:
            self.pos[0] -= dx
            self.pos[2] += dz
        if keys[key.A]:
            self.pos[0] -= dz
            self.pos[2] -= dx
        if keys[key.D]:
            self.pos[0] += dz
            self.pos[2] += dx

        if keys[key.SPACE]:
            self.pos[1] += s
        if keys[key.LSHIFT]:
            self.pos[1] -= s


class Window(pyglet.window.Window):
    def push(self, pos, rot):
        '''
        Camera align
        '''
        glPushMatrix()
        glRotatef(-rot[0], 1, 0, 0)
        glRotatef(-rot[1], 0, 1, 0)
        glTranslatef(-pos[0], -pos[1], -pos[2],)

    def Projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

    def Model(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def set2d(self):
        self.Projection()
        gluOrtho2D(0, self.width, 0, self.height)
        self.Model()

    def set3d(self):
        self.Projection()
        gluPerspective(70, self.width/self.height, 0.05, 1000)
        self.Model()

    def setLock(self, state):
        self.lock = state

    lock = False
    mouse_lock = property(lambda self: self.lock, setLock)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_minimum_size(300, 200)
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        pyglet.clock.schedule(self.update)

        self.meshes = [Block()]
        self.camera = Player((0.5, 1.5, 1.5), (-30, 0))

    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            self.mouse_lock = not self.mouse_lock

    def on_mouse_release(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            self.mouse_lock = not self.mouse_lock

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons == mouse.LEFT:
            self.camera.mouse_motion(dx, dy)
        if buttons == mouse.MIDDLE:
            self.camera.pos[0] -= dx / 16
            self.camera.pos[2] += dy / 16

    def on_key_press(self, KEY, MOD):
        if KEY == key.ESCAPE:
            self.close()
        elif KEY == key.E:
            self.mouse_lock = not self.mouse_lock

    def update(self, dt):
        self.camera.update(dt, self.keys)

    def on_draw(self):
        self.clear()
        self.set3d()
        self.push(self.camera.pos, self.camera.rot)
        
        for mesh in self.meshes:
            mesh.draw()
        
        glPopMatrix()


window = Window(width=854, height=480, caption='Minecraft', resizable=True)
glClearColor(0.5, 0.7, 1, 1)
glEnable(GL_DEPTH_TEST)
# glEnable(GL_CULL_FACE)
pyglet.app.run()
