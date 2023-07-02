import taichi as ti
from taichi.math import cmul, dot, log2, vec2, vec3
ti.init(arch=ti.cpu)

width, height = 800, 640
pixels = ti.Vector.field(3, float, shape=(width, height))

@ti.func
def setcolor(z, i):
    v = log2(i + 1 - log2(log2(z.norm()))) / 5
    col = vec3(0)
    if v < 1.0:
        col = vec3(v**4, v**2.5, v)
    else:
        v = ti.max(0., 2 - v)
        col = vec3(v, v**1.5, v**3)
    return col

@ti.kernel
def render():
    for i, j in pixels:
        c = 2.0 * vec2(i, j) / height - vec2(1.8, 1)
        z = vec2(0)
        count = 0
        while count < 100 and dot(z, z) < 50:
            z = cmul(z, z) + c
            count += 1

        if count == 100:
            pixels[i, j] = [0, 0, 0]
        else:
            pixels[i, j] = setcolor(z, count)

render()
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

@ti.func
def complex_sqr(z):  # complex square of a 2D vector
    return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])

@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
        c = tm.vec2(-0.8, tm.cos(t) * 0.2)
        z = tm.vec2(i / n - 1, j / n - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

gui = ti.GUI("Julia Set", res=(n * 2, n))

i = 0
while gui.running:
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
    i += 1