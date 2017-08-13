##DeepLearning##
> 收集了关于深度学习的一些学习资料，主要包括CNN,RNN,GAN的论文和网课，模型的具体实现可以参考Tensorflow的[models][1],里面包括了很多深度学习的模型实现．

###CNN###
> 在ImageNet比赛中出现的比较重要的CNN结构
> [AlexNet][2]--2012年ILSVRC的优胜者
> [VGG][3]--Very Deep Convolutional Networks for Large-Scale Image Recognition,主要专注于加深网络结构从而提升网络性能
> [Network in Network][4]--新的卷积结构，Inception的基础，卷积结构的复杂带来性能提升
> [GoogLeNet][5]--Inception v1,2014年ILSVRC
> [Inception v2,Inception v3][6]--进一步改进GoogLeNet的卷积结构
> [Inception v4][7]--Inception v4和Inception-ResNet
> [ResNet][8]--残差网络，使得非常深的CNN可以有效的训练，不会出现性能退化的情况
> [Identity Mappings in ResNet][9]--对于ResNet结构的改进

###GAN###
> [GAN][10]--GAN网络的出现
> [DCGAN][11]--CNN与GAN的结合
> [WGAN][12]--改进GAN难以训练的问题
> [CGAN][13]--利用Conditional Adversarial Networks进行图像风格的转换

[1]:https://github.com/tensorflow/models
[2]:https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
[3]:https://arxiv.org/pdf/1409.1556.pdf
[4]:https://arxiv.org/pdf/1312.4400v3.pdf
[5]:https://arxiv.org/pdf/1409.4842v1.pdf
[6]:https://arxiv.org/pdf/1512.00567v3.pdf
[7]:https://arxiv.org/pdf/1602.07261v2.pdf
[8]:https://arxiv.org/pdf/1512.03385.pdf
[9]:https://arxiv.org/pdf/1603.05027.pdf
[10]:https://arxiv.org/pdf/1406.2661.pdf
[11]:https://arxiv.org/pdf/1511.06434.pdf
[12]:https://arxiv.org/pdf/1701.07875.pdf
[13]:https://arxiv.org/pdf/1611.07004.pdf
