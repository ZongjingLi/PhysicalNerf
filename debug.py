import taichi as ti
import taichi.math as tm

ti.init(arch = ti.gpu)

n = 320
pixels = ti.field(dtype = float, shape = (2*n, n))

@ti.func
def complex_sqr(z):
    return tm.vec2(z[0]*z[0] - z[1]*z[1], 2*z[0]*z[1])

@ti.kernel
def paint(t: float):
    for x,y in pixels:
        c = tm.vec2(-.8, tm.cos(t) * .2)
        z = tm.vec2(x / n - 1, y / n - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            iterations += 1
            z = complex_sqr(z) + c
        pixels[x,y] = 1 - iterations * 0.02

gui = ti.GUI("test_run", res = (2*n, n))

t = 0
while gui.running:
    paint(t * 0.03)
    gui.set_image(pixels)
    gui.show()
    t += 1