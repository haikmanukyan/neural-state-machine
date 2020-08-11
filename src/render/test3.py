import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
 
 
def main():
    if not glfw.init():
        return
 
    window = glfw.create_window(720, 600, "Pyopengl Rotating Cube", None, None)
 
    if not window:
        glfw.terminate()
        return
 
    glfw.make_context_current(window)
 
    cube = [-0.5, -0.5, 0.5, 1.0, 0.0, 0.0,
            0.5, -0.5, 0.5, 0.0, 1.0, 0.0,
            0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
            -0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
 
            -0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
            0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
            0.5, 0.5, -0.5, 0.0, 0.0, 1.0,
            -0.5, 0.5, -0.5, 1.0, 1.0, 1.0]
 
    # convert to 32bit float
 
 
    cube = np.array(cube, dtype=np.float32)
 
    indices = [0, 1, 2, 2, 3, 0,
               4, 5, 6, 6, 7, 4,
               4, 5, 1, 1, 0, 4,
               6, 7, 3, 3, 2, 6,
               5, 6, 2, 2, 1, 5,
               7, 4, 0, 0, 3, 7]
 
    indices = np.array(indices, dtype = np.uint32)
 
 
 
    VERTEX_SHADER = """
 
        #version 330
 
        in vec3 position;
        in vec3 color;
        out vec3 newColor;
        
        uniform mat4 transform; 
 
        void main() {
 
         gl_Position = transform * vec4(position, 1.0f);
         newColor = color;
 
          }
 
 
    """
 
    FRAGMENT_SHADER = """
        #version 330
 
        in vec3 newColor;
        out vec4 outColor;
 
        void main() {
 
          outColor = vec4(newColor, 1.0f);
 
        }
 
    """
 
    # Compile The Program and shaders
 
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
 
    # Create Buffer object in gpu
    VBO = glGenBuffers(1)
    # Bind the buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 192, cube, GL_STATIC_DRAW)
 
    #Create EBO
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 144, indices, GL_STATIC_DRAW)
 
 
 
 
    # get the position from  shader
    position = glGetAttribLocation(shader, 'position')
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)
 
    # get the color from  shader
    color = glGetAttribLocation(shader, 'color')
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)
 
    glUseProgram(shader)
 
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)
 
    while not glfw.window_should_close(window):
        glfw.poll_events()
 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
 
        rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
        rot_y = pyrr.Matrix44.from_y_rotation(0.8 * glfw.get_time())
 
        transformLoc = glGetUniformLocation(shader, "transform")
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)
 
        # Draw Cube
 
        glDrawElements(GL_TRIANGLES,36, GL_UNSIGNED_INT,  None)
 
        glfw.swap_buffers(window)
 
    glfw.terminate()
 
 
if __name__ == "__main__":
    main()