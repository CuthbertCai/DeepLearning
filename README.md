## DeepLearning ##
> 收集了关于深度学习的一些学习资料，主要包括CNN,RNN,GAN的论文和网课，模型的具体实现可以参考Tensorflow的[models][1],里面包括了很多深度学习的模型实现．

### CNN ###
> #### 在ImageNet比赛中出现的比较重要的CNN结构以及ResNet的各种变体  
> [AlexNet][2]--2012年ILSVRC的优胜者  
> [VGG][3]--Very Deep Convolutional Networks for Large-Scale Image Recognition,主要专注于加深网络结构从而提升网络性能  
> [Network in Network][4]--新的卷积结构，Inception的基础，卷积结构的复杂带来性能提升  
> [GoogLeNet][5]--Inception v1,2014年ILSVRC  
> [Inception v2,Inception v3][6]--进一步改进GoogLeNet的卷积结构  
> [Inception v4][7]--Inception v4和Inception-ResNet  
> [Xception][20]--Inception的分析以及depthwise separable convolution  
> [ResNet][8]--残差网络，使得非常深的CNN可以有效的训练，不会出现性能退化的情况   
> [Identity Mappings in ResNet][9]--对于ResNet结构的改进  
> [Wide ResNet][14]--从宽度改进ResNet  
> [ResNeXt][15]--ResNet变体，改进卷积结构  
> [DenseNet][16]--ResNet变体，每一层都能得到之前所有层的输出  
> [DPN][19]--2017年ILSVRC，Dual Path Network  

> #### 一些精简模型，减少参数并保持准确率，可用于嵌入式设备  
> [MobileNet][17]--ResNet变体，应用于移动端的网络结构，大规模使用depthwise separable convolution  
> [ShuffleNet][18]--应用于移动端的ResNet,使用pointwise group convolution和channel shuffle  
> [SqueezeNet][21]--small DNN architecture,参数大大减少  

### GAN ###
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
[14]:https://arxiv.org/pdf/1605.07146.pdf
[15]:https://arxiv.org/pdf/1611.05431.pdf
[16]:https://arxiv.org/pdf/1608.06993.pdf
[17]:https://arxiv.org/pdf/1704.04861.pdf
[18]:https://arxiv.org/pdf/1707.01083.pdf
[19]:https://arxiv.org/pdf/1707.01629.pdf
[20]:https://arxiv.org/pdf/1610.02357.pdf
[21]:https://arxiv.org/pdf/1602.07360.pdf
