import torch
import torch.nn as nn
import taichi as ti



class VanillaNERF(nn.Module):
 
 
 
 
    def __init__(self, config):
        super().__init__()
        self.mlp = 0




    def forward(self, x, c2w):       




        return x


""""
###############################
model = init_model()


optim = torch.optim.Adam(model.parameters(),lr=5e-4)

N_samples = 64
epochs = 1000
psnrs = []
iternums = []
i_plot = 25



import time
t = time.time()
for i in range(epochs):
    
    img_i = np.random.randint(images.shape[0])
    target = images[img_i]
    pose = poses[img_i]
    rays_o, rays_d = get_rays(H, W, focal, pose)
    with tf.GradientTape() as tape:
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., 
N_samples=N_samples, rand=True)
        loss = tf.reduce_mean(tf.square(rgb - target))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if i%i_plot==0:
        print(i, (time.time() - t) / i_plot, 'secs per iter')
        t = time.time()
        
        # Render the holdout view for logging
        rays_o, rays_d = get_rays(H, W, focal, testpose)
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., 
N_samples=N_samples)
        loss = tf.reduce_mean(tf.square(rgb - testimg))
        psnr = -10. * tf.math.log(loss) / tf.math.log(10.)

        psnrs.append(psnr.numpy())
        iternums.append(i)
        
        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.imshow(rgb)
        plt.title(f'Iteration: {i}')
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title('PSNR')
        plt.show()

print('Done')

















####################

class PhysicsNerf(nn.Module):
    def __init__(self, config):
        super().__init__()











    def forward(self, x, c2w):return x




"""