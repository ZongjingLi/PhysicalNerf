import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model(t3.Mesh.from_obj(t3.readobj('assets/torus.obj', scale=0.8)))
model.material = t3.Material(t3.CookTorrance(
    color=t3.Texture(ti.imread('assets/cloth.jpg')),
    roughness=t3.Texture(ti.imread('assets/pattern.jpg')),
    metallic=t3.Constant(0.5),
    ))
scene.add_model(model)
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[0.8, 0, 2.5])
scene.add_camera(camera)
light = t3.Light([0, -0.5, -1])
scene.add_light(light)
ambient = t3.AmbientLight(0.3)
scene.add_light(ambient)

gui = ti.GUI('Model', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    model.L2W[None] = t3.rotateX(angle=t3.get_time())
    scene.render()
    gui.set_image(camera.img)
    gui.show()