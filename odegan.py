"""
Modified from https://github.com/pytorch/examples/blob/master/dcgan/main.py
"""
from __future__ import print_function
import argparse
import os
import random
import copy
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=False, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='./images/', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

    # ODE Params
    parser.add_argument('--ode', default='heun', choices=['heun', 'rk4'], help='Type of ode step to take')
    parser.add_argument('--step_size', type=float, default=0.01, help='Fixed step optimizer step size')
    parser.add_argument('--disc_reg', default=0.01, type=float,
                        help='Fixed weight decay of theta (discriminator)')

    opt = parser.parse_args()
    print(opt)

    opt.outf = os.path.join(opt.outf, opt.dataset + "_" + opt.ode)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(opt.outf, 'logs', timestamp)

    writer = SummaryWriter(log_dir=logdir)

    try:
        os.makedirs(opt.outf, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
        raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        nc = 3
    elif opt.dataset == 'lsun':
        classes = [c + '_train' for c in opt.classes.split(',')]
        dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc = 3
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc = 3

    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(opt.imageSize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
        nc = 1

    elif opt.dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                                transform=transforms.ToTensor())
        nc = 3

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)

    # custom weights initialization called on netG and netD
    # def weights_init(m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         torch.nn.init.normal_(m.weight, 0.0, 0.02)
    #     elif classname.find('BatchNorm') != -1:
    #         torch.nn.init.normal_(m.weight, 1.0, 0.02)
    #         torch.nn.init.zeros_(m.bias)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)


    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.project = nn.Conv2d(nz, ngf * 8 * 4 * 4, 1, 1, 0, bias=False)
            self.main = nn.Sequential(
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # # state size. (ngf) x 32 x 32
                nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                raise NotImplemented()
                # output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                x = self.project(input)
                x = x.view(-1, ngf * 8, 4, 4)
                output = self.main(x)

            return output


    # class Generator(nn.Module):
    #     def __init__(self, ngpu):
    #         super(Generator, self).__init__()
    #         self.ngpu = ngpu
    #         self.project = nn.Conv2d(nz, ngf * 4 * 4, 1, 1, 0, bias=False)
    #         self.main = nn.Sequential(
    #             # state size. (ngf*8) x 4 x 4
    #             nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
    #             nn.BatchNorm2d(ngf),
    #             nn.ReLU(True),
    #             # state size. (ngf*4) x 8 x 8
    #             nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
    #             nn.BatchNorm2d(ngf),
    #             nn.ReLU(True),
    #             # state size. (ngf*2) x 16 x 16
    #             nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
    #             nn.BatchNorm2d(ngf),
    #             nn.ReLU(True),
    #             # # state size. (ngf) x 32 x 32
    #             nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
    #             nn.Tanh()
    #             # state size. (nc) x 64 x 64
    #         )
    #
    #     def forward(self, input):
    #         if input.is_cuda and self.ngpu > 1:
    #             raise NotImplemented()
    #             # output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    #         else:
    #             x = self.project(input)
    #             x = x.view(x.shape[0], ngf, 4, 4)
    #             output = self.main(x)
    #
    #         return output


    # ODE GAN
    netG = Generator(ngpu)
    netG.apply(weights_init)
    netG = netG.to(device)

    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf) x 16 x 16
                nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*2) x 8 x 8
                nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*4) x 4 x 4
                nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*8) x 2 x 2
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                # nn.Sigmoid()
            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)

            return output.view(-1, 1).squeeze(1)


    # class Discriminator(nn.Module):
    #     def __init__(self, ngpu):
    #         super(Discriminator, self).__init__()
    #         self.ngpu = ngpu
    #         self.main = nn.Sequential(
    #             # input is (nc) x 32 x 32
    #             # block 0
    #             nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
    #             nn.BatchNorm2d(ndf),
    #             nn.LeakyReLU(0.1, inplace=True),
    #             nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
    #             nn.AvgPool2d(2),
    #             # state size. (ndf) x 16 x 16
    #             # block 1
    #             nn.LeakyReLU(0.1, inplace=True),
    #             nn.BatchNorm2d(ndf),
    #             nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
    #             nn.LeakyReLU(0.1, inplace=True),
    #             nn.BatchNorm2d(ndf),
    #             nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
    #             nn.AvgPool2d(2),
    #             # state size. (ndf*2) x 8 x 8
    #             # block 2
    #             nn.LeakyReLU(0.1, inplace=True),
    #             nn.BatchNorm2d(ndf),
    #             nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
    #             nn.LeakyReLU(0.1, inplace=True),
    #             nn.BatchNorm2d(ndf),
    #             nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
    #             # block 3
    #             nn.LeakyReLU(0.1, inplace=True),
    #             nn.BatchNorm2d(ndf),
    #             nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
    #             nn.LeakyReLU(0.1, inplace=True),
    #             nn.BatchNorm2d(ndf),
    #             nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
    #             # Global average pooling
    #             nn.LeakyReLU(0.1, inplace=True),
    #             nn.AdaptiveAvgPool2d(1),
    #             # state size. (ndf*4) x 1 x 1
    #             nn.Conv2d(ndf, 1, 1, 1, 0, bias=False),
    #             # nn.Sigmoid()
    #         )
    #
    #     def forward(self, input):
    #         if input.is_cuda and self.ngpu > 1:
    #             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    #         else:
    #             output = self.main(input)
    #
    #         return output.view(-1, 1).squeeze(1)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)

    netD = netD.to(device)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    if opt.dry_run:
        opt.niter = 1


    def grad_clone(source: torch.nn.Module):
        dest = copy.deepcopy(source)
        dest.requires_grad_(True)
        for s_p, d_p in zip(source.parameters(), dest.parameters()):
            if s_p.grad is not None:
                d_p.grad = s_p.grad.clone()

        return dest


    def normalize_grad(grad):
        # normalize gradient
        grad_norm = grad.norm()
        if grad_norm > 1.:
            grad.div_(grad_norm)
        return grad

    def heun_ode_step(G: Generator, D: Discriminator, data: torch.Tensor, step_size: float, disc_reg: float):
        # Compute first step of Heun
        theta_1, phi_1, errD, errG, D_x, D_G_z1, D_G_z2 = gan_step(G, D, data, detach_err=False, retain_graph=True)

        # Compute the L2 norm using the prior computation graph
        grad_norm = None  # errG
        for phi_0_param in G.parameters():
            if phi_0_param.grad is not None:
                if grad_norm is None:
                    # grad_norm = disc_reg * phi_0_param.grad.square().sum()
                    grad_norm = phi_0_param.grad.square().sum()
                else:
                    # grad_norm = grad_norm + disc_reg * phi_0_param.grad.square().sum()
                    grad_norm = grad_norm + phi_0_param.grad.square().sum()

        # print("grad norm", grad_norm)
        # grad_norm = disc_reg * grad_norm
        grad_norm = grad_norm.sqrt()

        # Preserve gradients for regularization in cache
        D_norm_grads = torch.autograd.grad(grad_norm, list(D.parameters()))
        grad_norm = grad_norm.detach()

        # Preserve the gradients of the discriminator gradients (obtained via the norm calculation)
        disc_grad_norm = torch.tensor(0.0, device=device)
        for d_grad, in zip(D_norm_grads):
            # compute discriminator norm
            disc_grad_norm = disc_grad_norm + d_grad.detach().square().sum().sqrt()

        # Detach graph
        errD = errD.detach()
        errG = errG.detach()

        # preserve theta, phi for next computation
        theta_0 = grad_clone(theta_1)
        phi_0 = grad_clone(phi_1)

        # Update theta and phi for first heun step]
        for d_param, theta_1_param in zip(D.parameters(), theta_1.parameters()):
            if theta_1_param.grad is not None:
                theta_1_param.data = d_param.data + (step_size * -theta_1_param.grad)

        for g_param, phi_1_param in zip(G.parameters(), phi_1.parameters()):
            if phi_1_param.grad is not None:
                phi_1_param.data = g_param.data + (step_size * -phi_1_param.grad)

        # Compute second step of Heun
        theta_2, phi_2, errD, errG, D_x, D_G_z1, D_G_z2 = gan_step(phi_1, theta_1, data)

        # Compute grad norm and update discriminator
        for d_param, theta_0_param, theta_1_param in zip(D.parameters(), theta_0.parameters(), theta_2.parameters()):
            if theta_1_param.grad is not None:
                grad = theta_0_param.grad + theta_1_param.grad

                # simulate regularization with weight decay
                # if disc_reg > 0:
                #     grad += disc_reg * d_param.data

                # normalize gradient
                grad = normalize_grad(grad)

                d_param.data = d_param.data + (step_size * 0.5 * -(grad))

        for g_param, phi_0_param, phi_1_param in zip(G.parameters(), phi_0.parameters(), phi_2.parameters()):
            if phi_1_param.grad is not None:
                grad = phi_0_param.grad + phi_1_param.grad

                # normalize gradient
                grad = normalize_grad(grad)

                g_param.data = g_param.data + (step_size * 0.5 * -(grad))

        # Regularization step
        for d_param, d_grad in zip(D.parameters(), D_norm_grads):
            # print("abs diff", d_param.data.abs().mean(), (step_size * disc_reg * d_grad).abs().mean())
            d_param.data = d_param.data - step_size * disc_reg * d_grad

        del theta_0, theta_1, theta_2
        del phi_0, phi_1, phi_2
        del D_norm_grads

        return G, D, errD, errG, D_x, D_G_z1, D_G_z2, grad_norm.detach(), disc_grad_norm.detach()


    def rk4_ode_step(G: Generator, D: Discriminator, data: torch.Tensor, step_size: float, disc_reg: float):
        # Compute first step of RK4
        theta_1_cache, phi_1_cache, errD, errG, D_x, D_G_z1, D_G_z2 = gan_step(G, D, data,
                                                                               detach_err=False, retain_graph=True)

        # Compute the L2 norm using the prior computation graph
        grad_norm = None  # errG
        for phi_0_param in G.parameters():
            if phi_0_param.grad is not None:
                if grad_norm is None:
                    # grad_norm = disc_reg * phi_0_param.grad.square().sum()
                    grad_norm = phi_0_param.grad.square().sum()
                else:
                    # grad_norm = grad_norm + disc_reg * phi_0_param.grad.square().sum()
                    grad_norm = grad_norm + phi_0_param.grad.square().sum()

        # print("grad norm", grad_norm)
        # grad_norm = disc_reg * grad_norm
        grad_norm = grad_norm.sqrt()

        # Preserve gradients for regularization in cache
        D_norm_grads = torch.autograd.grad(grad_norm, list(D.parameters()))
        grad_norm = grad_norm.detach()

        # Preserve the gradients of the discriminator gradients (obtained via the norm calculation)
        disc_grad_norm = torch.tensor(0.0, device=device)
        for d_grad, in zip(D_norm_grads):
            # compute discriminator norm
            disc_grad_norm = disc_grad_norm + d_grad.detach().square().sum().sqrt()

        # Detach graph
        errD = errD.detach()
        errG = errG.detach()

        # preserve theta1, phi1 for next computation
        theta_1 = grad_clone(theta_1_cache)
        phi_1 = grad_clone(phi_1_cache)

        # Update theta and phi for second RK step]
        for d_param, theta_1_param in zip(D.parameters(), theta_1.parameters()):
            if theta_1_param.grad is not None:
                theta_1_param.data = d_param.data + (step_size * 0.5 * -theta_1_param.grad)

        for g_param, phi_1_param in zip(G.parameters(), phi_1.parameters()):
            if phi_1_param.grad is not None:
                phi_1_param.data = g_param.data + (step_size * 0.5 * -phi_1_param.grad)

        # Compute second step of RK 4
        theta_2_cache, phi_2_cache, errD, errG, D_x, D_G_z1, D_G_z2 = gan_step(phi_1, theta_1, data)

        # preserve theta2, phi2
        theta_2 = grad_clone(theta_2_cache)
        phi_2 = grad_clone(phi_2_cache)

        # Update theta and phi for third RK step]
        for d_param, theta_2_param in zip(D.parameters(), theta_2.parameters()):
            if theta_2_param.grad is not None:
                theta_2_param.data = d_param.data + (step_size * 0.5 * -theta_2_param.grad)

        for g_param, phi_2_param in zip(G.parameters(), phi_2.parameters()):
            if phi_2_param.grad is not None:
                phi_2_param.data = g_param.data + (step_size * 0.5 * -phi_2_param.grad)

        # Compute third step of RK 4
        theta_3_cache, phi_3_cache, errD, errG, D_x, D_G_z1, D_G_z2 = gan_step(phi_2, theta_2, data)

        # preserve theta3, phi3
        theta_3 = grad_clone(theta_3_cache)
        phi_3 = grad_clone(phi_3_cache)

        # Update theta and phi for fourth RK step]
        for d_param, theta_3_param in zip(D.parameters(), theta_3.parameters()):
            if theta_3_param.grad is not None:
                theta_3_param.data = d_param.data + (step_size * -theta_3_param.grad)

        for g_param, phi_3_param in zip(G.parameters(), phi_3.parameters()):
            if phi_3_param.grad is not None:
                phi_3_param.data = g_param.data + (step_size * -phi_3_param.grad)

        # Compute fourth step of RK 4
        theta_4, phi_4, errD, errG, D_x, D_G_z1, D_G_z2 = gan_step(phi_3, theta_3, data)

        # Compute grad norm and update discriminator
        for d_param, theta_1_param, theta_2_param, theta_3_param, theta_4_param in zip(D.parameters(),
                                                                                       theta_1_cache.parameters(),
                                                                                       theta_2_cache.parameters(),
                                                                                       theta_3_cache.parameters(),
                                                                                       theta_4.parameters()):
            if theta_1_param.grad is not None:
                grad = (theta_1_param.grad + 2 * theta_2_param.grad + 2 * theta_3_param.grad + theta_4_param.grad)

                # simulate regularization with weight decay
                # if disc_reg > 0:
                #     grad += disc_reg * d_param.data

                # normalize gradient
                grad = normalize_grad(grad)

                d_param.data = d_param.data + (step_size / 6. * -(grad))

        for g_param, phi_1_param, phi_2_param, phi_3_param, phi_4_param in zip(G.parameters(),
                                                                               phi_1_cache.parameters(),
                                                                               phi_2_cache.parameters(),
                                                                               phi_3_cache.parameters(),
                                                                               phi_4.parameters()):
            if phi_1_param.grad is not None:
                grad = (phi_1_param.grad + 2 * phi_2_param.grad + 2 * phi_3_param.grad + phi_4_param.grad)

                # normalize gradient
                grad = normalize_grad(grad)

                g_param.data = g_param.data + (step_size / 6.0 * -(grad))

        # Regularization step
        for d_param, d_grad in zip(D.parameters(), D_norm_grads):
            if d_param.grad is not None:
                d_param.data = d_param.data - step_size * disc_reg * d_grad

        del theta_1, theta_1_cache, theta_2, theta_2_cache, theta_3, theta_3_cache, theta_4
        del phi_1, phi_1_cache, phi_2, phi_2_cache, phi_3, phi_3_cache, phi_4
        del D_norm_grads

        return G, D, errD, errG, D_x, D_G_z1, D_G_z2, grad_norm.detach(), disc_grad_norm.detach()


    def gan_step(G: Generator, D: Discriminator, data, detach_err: bool = True, retain_graph: bool = False) -> (
            Discriminator, Generator, torch.Tensor, torch.Tensor, torch.Tensor):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        D.zero_grad()

        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)

        output = D(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().detach()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = G(noise)
        label.fill_(fake_label)
        output = D(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().detach()
        errD = errD_real + errD_fake

        if detach_err:
            errD = errD.detach()

        DISC_GRAD_CACHE = grad_clone(D)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        G.zero_grad()

        label.fill_(real_label)  # fake labels are real for generator cost
        output = D(fake)
        errG = criterion(output, label)
        errG.backward(create_graph=retain_graph)
        D_G_z2 = output.mean().detach()

        if detach_err:
            errG = errG.detach()

        GEN_GRAD_CACHE = grad_clone(G)

        return DISC_GRAD_CACHE, GEN_GRAD_CACHE, errD, errG, D_x, D_G_z1, D_G_z2

    # Save hyper parameters
    writer.add_hparams(vars(opt), metric_dict={})

    step_size = opt.step_size
    global_step = 0

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            # Schedule
            if epoch == 0 and i == 500:
                step_size = opt.step_size * 4
            elif epoch == 100 and i == 0:
                step_size = opt.step_size * 2
            elif epoch == 180 and i == 0:
                step_size = opt.step_size

            if opt.ode == 'heun':

                netG, netD, errD, errG, D_x, D_G_z1, D_G_z2, gen_grad_norm, disc_grad_norm = heun_ode_step(netG, netD,
                                                                                                           data,
                                                                                                           step_size=step_size,
                                                                                                           disc_reg=opt.disc_reg)

            else:
                netG, netD, errD, errG, D_x, D_G_z1, D_G_z2, gen_grad_norm, disc_grad_norm = rk4_ode_step(netG, netD,
                                                                                                          data,
                                                                                                          step_size=step_size,
                                                                                                          disc_reg=opt.disc_reg)

            # Cast logits to sigmoid probabilities
            D_x = D_x.sigmoid().item()
            D_G_z1 = D_G_z1.sigmoid().item()
            D_G_z2 = D_G_z2.sigmoid().item()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f '
                  'Gen Grad Norm: %0.4f Disc Grad Norm: %0.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, gen_grad_norm, disc_grad_norm))

            writer.add_scalar('loss/discriminator', errD.item(), global_step=global_step)
            writer.add_scalar('loss/generator', errG.item(), global_step=global_step)
            writer.add_scalar('acc/D(x)', D_x, global_step=global_step)
            writer.add_scalar('acc/D(G(z))-fake', D_G_z1, global_step=global_step)
            writer.add_scalar('acc/D(G(z))-real', D_G_z2, global_step=global_step)
            writer.add_scalar('norm/gen_grad_norm', gen_grad_norm, global_step=global_step)
            writer.add_scalar('norm/disc_grad_norm', disc_grad_norm, global_step=global_step)
            writer.add_scalar('step_size', step_size, global_step=global_step)

            global_step += 1

            if i % 100 == 0:
                real_cpu = data[0].to(device)
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % opt.outf,
                                  normalize=True)

                # fake = netG(fixed_noise)

                random_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
                fake = netG(random_noise)

                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                                  normalize=True)

            if opt.dry_run:
                break
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

    writer.flush()
