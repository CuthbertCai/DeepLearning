## DeepLearning ##
> 收集了关于深度学习的一些学习资料，主要包括CNN,RNN,GAN的论文和网课，模型的具体实现可以参考Tensorflow的[models][1],里面包括了很多深度学习的模型实现．

### CNN ###
> #### [A Guide to Convolution Arithmetic for Deep Learning][37]  
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

> #### CNN for Object Detection ####
> [R-CNN][27]--Regions with CNN features  
> [Fast R-CNN][28]--Fast Region-based Convolutional Network method, using spatial pramid pooling  
> [Faster R-CNN][29]--Real-Time Object Dectection with Region Proposal Networks  
> [SPP-Net][30]--Spatial Pyramid Pooling, can generate fixed-length representation regardless of image size    

### RNN ###
> [Understanding LSTM Networks][24]:介绍RNN和LSTM的博客，可作为RNN和LSTM入门的学习资料  
> [GRU][25]:A variation of LSTM, Gated Recurrent Unit  
> [Attention and Augmented Recurrent Neural Networks][26]:A blog about some variations of RNN, especially attention.  

### GAN ###
> [GAN][10]--GAN网络的出现  
> [DCGAN][11]--CNN与GAN的结合  
> [WGAN][12]--改进GAN难以训练的问题  
> [CGAN][13]--利用Conditional Adversarial Networks进行图像风格的转换  

### CVPR 2017 ###
> [FRRN][33]--Full-Resolution Residual Networks, one stream for recogniton while the other stream for localization, without pre-train  
> [PointNet][34]--Deep learning on Points Sets for Classification and Segmentation, derectly consuming points sets, robust woth respect to input perturbation, unified structure for different tasks  
> [SRGAN][35]--Single Image Super-Resolution using GAN, perceptual loss for high quality  
> [Deep Image Matting][36]--A CNN encoder-decoder followed a small ResNet, an end-to-end model with a large scale dataset they create  
> [Deep Reinforcement Learning-based Image Captioning][38]--Decsion-making framework, a policy network, a value network and reward defined by visual-semantic embedding   
> [NOC][39]--Novel Object Captioner,  a network that can be trained end-to-end using a joint training strategy to integrate knowledge from external visual recognition datasets as well as semantic information from independent unannotated text corpora to generate captions for a diverse range of rare and novel objects  
> [DeliGAN][40]--GAN for Diverse and Limited Data, modify the distrubution of latent space and design a modified version of 'Inception Score' to quantitatively characterize the diversity of generated samples  
> [SCA-CNN][41]--Spatial and Channel-Wise Attention in CNN for Image Captioning, dynamically modulated the sentence generation context in multi-layer feature maps with spatial and channel-wise attention  
> [Realtime Multi-Person 2D Pose Estimation using PAFs][42]--One branch CNN for parts detection which produces confidence maps, the other branch CNN for part association with part affinity fields  
> [CDC][43]--Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos, C3D followed CDC filter layers which performs convolution in space and deconvolution in time, and classifies acton for each frame  
> [DR-Net][44]--Detect Visual Relationships with Deep Relational Networks, exploit statistical dependencies between objects and their relationships, the statistical inference procedure is embedded into DR-Net  
> [Social Scene Understanding][45]--End-to-End Multi-Person Action Localization and Collective Activity Recognition, one FCN for extracting features, one DFCN for detecting persons and a RNN for outputing individual action and collective activity  

### Lectures ###
> [CS231n][22]--Convolutional Neural Networks for Visual Recognition,配有中文字幕的视频[点这][23]，需要翻墙  
> [CS224d][31]--Deep Learning for Natural Language Processing,[视频][32]  

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
[22]:http://cs231n.stanford.edu/
[23]:https://www.youtube.com/watch?v=NfnWJUyUJYU&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC
[24]:http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[25]:https://arxiv.org/pdf/1406.1078v3.pdf
[26]:https://distill.pub/2016/augmented-rnns/
[27]:https://arxiv.org/pdf/1311.2524v5.pdf
[28]:https://arxiv.org/pdf/1504.08083.pdf
[29]:https://arxiv.org/pdf/1506.01497v3.pdf
[30]:https://arxiv.org/pdf/1406.4729.pdf
[31]:http://cs224d.stanford.edu/
[32]:https://www.youtube.com/watch?v=ASn7ExxLZws&index=3&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6
[33]:https://arxiv.org/pdf/1611.08323.pdf
[34]:http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf
[35]:https://arxiv.org/pdf/1609.04802.pdf
[36]:https://arxiv.org/pdf/1703.03872.pdf
[37]:https://arxiv.org/pdf/1603.07285.pdf
[38]:http://openaccess.thecvf.com/content_cvpr_2017/papers/Ren_Deep_Reinforcement_Learning-Based_CVPR_2017_paper.pdf
[39]:http://openaccess.thecvf.com/content_cvpr_2017/papers/Venugopalan_Captioning_Images_With_CVPR_2017_paper.pdf
[40]:http://openaccess.thecvf.com/content_cvpr_2017/papers/Gurumurthy_DeLiGAN__Generative_CVPR_2017_paper.pdf
[41]:http://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_SCA-CNN_Spatial_and_CVPR_2017_paper.pdf
[42]:https://arxiv.org/pdf/1611.08050.pdf
[43]:https://arxiv.org/pdf/1703.01515.pdf
[44]:http://openaccess.thecvf.com/content_cvpr_2017/papers/Dai_Detecting_Visual_Relationships_CVPR_2017_paper.pdf
[45]:http://www.idiap.ch/~fleuret/papers/bagautdinov-et-al-cvpr2017.pdf
