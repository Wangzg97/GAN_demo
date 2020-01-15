# GAN_demo

## 简介
GAN的基本原理其实非常简单，这里以生成图片为例进行说明。假设我们有两个网络，G（Generator）和D（Discriminator）。正如它的名字所暗示的那样，它们的功能分别是：
G是一个生成图片的网络，它接收一个随机的噪声z，通过这个噪声生成图片，记做G(z)。
D是一个判别网络，判别一张图片是不是“真实的”。它的输入参数是x，x代表一张图片，输出D（x）代表x为真实图片的概率，如果为1，就代表100%是真实的图片，而输出为0，就代表不可能是真实的图片。
在训练过程中，生成网络G的目标就是尽量生成真实的图片去欺骗判别网络D。而D的目标就是尽量把G生成的图片和真实的图片分别开来。这样，G和D构成了一个动态的“博弈过程”。

## 项目简介：
- demo名称：利用生成式对抗网络生成动漫头像
- 环境：Windows10+Python3.7+Tensorflow2.0

## 文件介绍：
#### dataset.py：
- 主要是处理用来训练的动漫头像图片。
- 训练用的图片素材大小为[96, 96, 3]。[这里是链接](https://pan.baidu.com/s/17MuLkf35KEOhP7NSaPOL9w)。图片素材是在网上找的，忘记是哪位老哥的了，侵删。
- 图片示例![Snipaste_2020-01-15_15-00-00.png](https://i.loli.net/2020/01/15/A1yLOQhvjExcDzg.png)

#### main.py
简单的生成器与判别器
![Snipaste_2020-01-15_15-03-59.png](https://i.loli.net/2020/01/15/5zTYE672ZK1Mprx.png)

#### main2.py
- 生成器：UNet256
![Snipaste_2020-01-15_15-06-16.png](https://i.loli.net/2020/01/15/gvSpYCmWhH7sZ8J.png)
- 判别器：PatchGAN
![Snipaste_2020-01-15_15-06-27.png](https://i.loli.net/2020/01/15/PQhBqK683TLHysW.png)

#### final.py
- 生成器：考虑到生成器是从生成一个噪点开始的，因此UNet256中向下采样的过程没什么必要，所以去掉了向下采样的过程
![Snipaste_2020-01-15_15-09-25.png](https://i.loli.net/2020/01/15/yWpQNtvsA7bxgGD.png)
- 判别器：在PatchGAN基础上改造
![Snipaste_2020-01-15_15-09-37.png](https://i.loli.net/2020/01/15/AyvOfR6BwSxHj2V.png)
