# ODE GAN (Prototype) in PyTorch
Partial implementation of ODE-GAN technique from the paper [Training Generative Adversarial Networks by Solving Ordinary Differential Equations](https://arxiv.org/abs/2010.15040).

# Caveat
This is **not a faithful reproduction of the paper**! 

- One of the many major difference is the use of gradient normalization to stabilize training (and avoid exploding gradients which lead to nans in generator + discriminator).
- Another difference might be implementation of the regularization component. 
- Finally, this is a prototype to demonstrate the training regiment, without any focus for optimization of any kind - there's a lot of duplication of weights, caches etc throughout the code.

# Training Regiment
By default, the model is trained on the CIFAR 10 dataset, with most of the parameters set in argparse. 

Here is a tensorboard of a model being trained using RK2 (Heuns ODE step) for 250 epochs ~ 187500 update steps - [Tensorboard Dev Log](https://tensorboard.dev/experiment/E9VIqTYgT9umwIbiMVj33Q/#scalars&runSelectionState=eyIyMDIwLTExLTEwLTE3LTU1LTAxIjp0cnVlLCIyMDIwLTExLTEwLTE3LTU1LTAxXFwxNjA1MDU5NzA1LjkyNjM2NTEiOmZhbHNlfQ%3D%3D)

# Generated images
Training has not completed yet, here are images at the 60th epoch of training. Assuming nothing crashes in the next 200 epochs, there might be better results in later epochs.

<div align=center>
<img src="https://github.com/titu1994/pytorch_odegan/blob/master/data/fake_samples_epoch_060.png?raw=true" height=50% width=50%>
</div>
