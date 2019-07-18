# 真棒-Pytorch列表

[![pytorch车标暗](https://raw.githubusercontent.com/qbmzc/images/master/mdimage/2019/20190718144253849_468354148.png?token=AGMSQSURT3DLHUVNDFRCGHC5GALSG)](https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png)

[![](_v_images/20190718144250046_703455072)](https://camo.githubusercontent.com/23016a146059e3e548356f3efa16ff8c80c8e82d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f73746172732d373530302b2d627269676874677265656e2e7376673f7374796c653d666c6174) [![](_v_images/20190718144249045_1456287070)](https://camo.githubusercontent.com/9cfdf2774a70fb4b423fe3a427bfcd775c31ae2a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f666f726b732d313530302532422d627269676874677265656e2e737667) [![](_v_images/20190718144248143_159252973)](https://camo.githubusercontent.com/926d8ca67df15de5bd1abac234c0603d94f66c00/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f6e747269627574696f6e732d77656c636f6d652d627269676874677265656e2e7376673f7374796c653d666c6174)

## [](https://github.com/qbmzc/Awesome-pytorch-list#contents)内容

- [Pytorch及相关图书馆](https://github.com/qbmzc/Awesome-pytorch-list#pytorch--related-libraries)
    - [NLP和语音处理](https://github.com/qbmzc/Awesome-pytorch-list#nlp--Speech-Processing)
    - [计算机视觉](https://github.com/qbmzc/Awesome-pytorch-list#cv)
    - [概率/生成库](https://github.com/qbmzc/Awesome-pytorch-list#probabilisticgenerative-libraries)
    - [其他图书馆](https://github.com/qbmzc/Awesome-pytorch-list#other-libraries)
- [教程和示例](https://github.com/qbmzc/Awesome-pytorch-list#tutorials--examples)
- [论文实施](https://github.com/qbmzc/Awesome-pytorch-list#paper-implementations)
- [Pytorch在其他地方](https://github.com/qbmzc/Awesome-pytorch-list#pytorch-elsewhere)

## [](https://github.com/qbmzc/Awesome-pytorch-list#pytorch--related-libraries)Pytorch及相关图书馆

1. [pytorch](http://pytorch.org/)：Python中的张量和动态神经网络，具有强大的GPU加速功能。

### [](https://github.com/qbmzc/Awesome-pytorch-list#nlp--speech-processing)NLP和语音处理：

1. [pytorch text](https://github.com/pytorch/text)：火炬文本相关内容。
2. [pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)：[PyTorch中](https://github.com/IBM/pytorch-seq2seq)实现的序列到序列（seq2seq）模型的框架。
3. [anuvada](https://github.com/Sandeep42/anuvada)：使用PyTorch进行NLP的可解释模型。
4. [audio](https://github.com/pytorch/audio)：用于pytorch的简单音频I / O.
5. [loop](https://github.com/facebookresearch/loop)：一种跨多个扬声器生成语音的方法
6. [fairseq-py](https://github.com/facebookresearch/fairseq-py)：用Python编写的Facebook AI研究序列到序列工具包。
7. [演讲](https://github.com/awni/speech)：PyTorch ASR实施。
8. [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)：PyTorch [http://opennmt.net中](http://opennmt.net/)的开源神经机器翻译[](http://opennmt.net/)
9. [neuralcoref](https://github.com/huggingface/neuralcoref)：基于神经网络和spaCy huggingface.co/coref的最先进的共指消解
10. [情绪发现](https://github.com/NVIDIA/sentiment-discovery)：大规模的无监督语言建模，用于稳健的情绪分类。
11. [MUSE](https://github.com/facebookresearch/MUSE)：多语言无监督或监督词嵌入的库
12. [nmtpytorch](https://github.com/lium-lst/nmtpytorch)：[PyTorch中的](https://github.com/lium-lst/nmtpytorch)神经机器翻译框架。
13. [pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet)：WaveNet的快速生成实现
14. [Tacotron-pytorch](https://github.com/soobinseo/Tacotron-pytorch)：Tacotron：走向端到端语音合成。
15. [AllenNLP](https://github.com/allenai/allennlp)：一个基于PyTorch构建的开源NLP研究库。
16. [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP)：[PyTorch的](https://github.com/PetrochukM/PyTorch-NLP)文本实用程序和数据集pytorchnlp.readthedocs.io
17. [quick-nlp](https://github.com/outcastofmusic/quick-nlp)：基于FastAI的Pytorch NLP库。
18. [TTS](https://github.com/mozilla/TTS)：深入学习Text2Speech
19. [激光](https://github.com/facebookresearch/LASER)：语言不可知的能力表征
20. [pyannote-audio](https://github.com/pyannote/pyannote-audio)：用于说话人日记的神经构建块：语音活动检测，说话人变化检测，扬声器嵌入
21. [gensen](https://github.com/Maluuba/gensen)：通过大规模多任务学习学习通用分布式句子表示。
22. [翻译](https://github.com/pytorch/translate)：翻译 \- PyTorch语言库。
23. [espnet](https://github.com/espnet/espnet)：端到端语音处理工具包espnet.github.io/espnet
24. [pythia](https://github.com/facebookresearch/pythia)：Visual Question Answering的软件套件
25. [无监督的MT](https://github.com/facebookresearch/UnsupervisedMT)：基于短语和神经无监督的机器翻译。
26. [jiant](https://github.com/jsalt18-sentence-repl/jiant)：jiant语句表示学习工具包。
27. [BERT-PyTorch](https://github.com/codertimo/BERT-pytorch)：Pytorch实现了Google AI的2018 BERT，带有简单的注释
28. [InferSent](https://github.com/facebookresearch/InferSent)：句子嵌入（InferSent）和NLI的训练代码。
29. [uis-rnn](https://github.com/google/uis-rnn)：这是无界交错状态递归神经网络（UIS-RNN）算法的库，对应于纸质全监督扬声器二值化。arxiv.org/abs/1810.04719
30. [天赋](https://github.com/zalandoresearch/flair)：一个非常简单的框架，用于最先进的自然语言处理（NLP）
31. [pytext](https://github.com/facebookresearch/pytext)：基于PyTorch fb.me/pytextdocs的自然语言建模框架
32. [voicefilter](https://github.com/mindslab-ai/voicefilter)：Google AI的VoiceFilter系统的非官方PyTorch实现[http://swpark.me/voicefilter](http://swpark.me/voicefilter)
33. [BERT-NER](https://github.com/kamalkraj/BERT-NER)：Pytorch命名的实体识别与BERT。
34. [transfer-nlp](https://github.com/feedly/transfer-nlp)：NLP库，专为灵活的研究和开发而设计

### [](https://github.com/qbmzc/Awesome-pytorch-list#cv)简历：

1. [pytorch vision](https://github.com/pytorch/vision)：特定于Computer Vision的数据集，变换和模型。
2. [pt-styletransfer](https://github.com/tymokvo/pt-styletransfer)：神经风格转移作为PyTorch中的一个类。
3. [OpenFacePytorch](https://github.com/thnkim/OpenFacePytorch)：使用OpenFace的nn4.small2.v1.t7模型的PyTorch模块
4. [img\_classification\_pk_pytorch](https://github.com/felixgwu/img_classification_pk_pytorch)：快速将您的图像分类模型与最先进的模型（如DenseNet，ResNet，......）进行比较
5. [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)：子流形稀疏卷积网络。
6. [Convolution\_LSTM\_pytorch](https://github.com/automan000/Convolution_LSTM_pytorch)：多层卷积LSTM模块
7. [面部对齐](https://github.com/1adrianb/face-alignment)：![火](_v_images/20190718144246934_348033474.png) 使用pytorch adrianbulat.com构建2D和3D Face对齐库
8. [pytorch-semantic-segmentation](https://github.com/ZijunDeng/pytorch-semantic-segmentation)：用于语义分割的PyTorch。
9. [RoIAlign.pytorch](https://github.com/longcw/RoIAlign.pytorch)：这是RoIAlign的PyTorch版本。此实现基于crop\_and\_resize，并支持CPU和GPU上的前向和后向。
10. [pytorch-cnn-finetune](https://github.com/creafz/pytorch-cnn-finetune)：用PyTorch微调预训练卷积神经网络。
11. [detectorch](https://github.com/ignacio-rocco/detectorch)：检测器 \- 用于PyTorch的[检测](https://github.com/ignacio-rocco/detectorch)器
12. [Augmentor](https://github.com/mdbloice/Augmentor)：用于机器学习的Python图像增强库。[http://augmentor.readthedocs.io](http://augmentor.readthedocs.io/)
13. [s2cnn](https://github.com/jonas-koehler/s2cnn)：该库包含用于球形信号的SO（3）等变CNN的PyTorch实现（例如全向摄像机，地球上的信号）
14. [PyTorchCV](https://github.com/CVBox/PyTorchCV)：基于PyTorch的计算机视觉深度学习框架。
15. [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)：[PyTorch](https://github.com/facebookresearch/maskrcnn-benchmark)中实例分段和对象检测算法的快速模块化参考实现。
16. [image-classification-mobile](https://github.com/osmr/imgclsmob)：在ImageNet-1K上预训练的分类模型的集合。
17. [medicaltorch](https://github.com/perone/medicaltorch)：Pytorch的医学成像框架[http://medicaltorch.readthedocs.io](http://medicaltorch.readthedocs.io/)
18. [albumentations](https://github.com/albu/albumentations)：快速图像增强库。
19. [kornia](https://github.com/arraiyopensource/kornia)：可[分辨的](https://github.com/arraiyopensource/kornia)计算机视觉库。

### [](https://github.com/qbmzc/Awesome-pytorch-list#probabilisticgenerative-libraries)概率/生成库：

1. [ptstat](https://github.com/stepelu/ptstat)：[PyTorch中的](https://github.com/stepelu/ptstat)概率编程和统计推断
2. [pyro](https://github.com/uber/pyro)：使用Python和PyTorch进行深度通用概率编程[http://pyro.ai](http://pyro.ai/)
3. [probtorch](https://github.com/probtorch/probtorch)：Probabilistic Torch是扩展PyTorch的深度生成模型的库。
4. [paysage](https://github.com/drckf/paysage)：python / pytorch中的无监督学习和生成模型。
5. [pyvarinf](https://github.com/ctallec/pyvarinf)：Python包，便于使用贝叶斯深度学习方法和PyTorch的变分推理。
6. [pyprob](https://github.com/probprog/pyprob)：基于PyTorch的概率编程和推理编译库。
7. [mia](https://github.com/spring-epfl/mia)：用于运行针对ML模型的成员资格推断攻击的库。
8. [pro\_gan\_pytorch](https://github.com/akanimax/pro_gan_pytorch)：作为PyTorch nn.Module的扩展实现的ProGAN包。
9. [botorch](https://github.com/pytorch/botorch)：[PyTorch中的](https://github.com/pytorch/botorch)贝叶斯优化

### [](https://github.com/qbmzc/Awesome-pytorch-list#other-libraries)其他图书馆：

1. [pytorch extras](https://github.com/mrdrozdov/pytorch-extras)：[pytorch的](https://github.com/mrdrozdov/pytorch-extras)一些额外功能。
2. [功能动物园](https://github.com/szagoruyko/functional-zoo)：与lua torch不同，PyTorch在其核心中具有autograd，因此不需要使用torch.nn模块的模块化结构，可以轻松分配所需的变量并编写利用它们的函数，这有时更方便。此repo包含此功能方式的模型定义，某些模型具有预训练权重。
3. [torch-sampling](https://github.com/ncullen93/torchsample)：该软件包提供了一组转换和数据结构，用于从内存或内存不足的数据中进行采样。
4. [torchcraft-py](https://github.com/deepcraft/torchcraft-py)：[TorchCraft的](https://github.com/deepcraft/torchcraft-py) Python包装器，是Torch和星际争霸之间用于人工智能研究的桥梁。
5. [aorun](https://github.com/ramon-oliveira/aorun)：Aorun打算和PyTorch一起成为Keras的后端。
6. [记录器](https://github.com/oval-group/logger)：一个简单的实验记录器。
7. [PyTorch-docset](https://github.com/iamaziz/PyTorch-docset)：PyTorch docset！与Dash，Zeal，Velocity或LovelyDocs一起使用。
8. [convert\_torch\_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch)：将火炬t7模型转换为pytorch模型和源。
9. [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)：这个回购的目的是帮助重现研究论文的结果。
10. [pytorch_fft](https://github.com/locuslab/pytorch_fft)：用于FFT的PyTorch包装器
11. [caffe\_to\_torch\_to\_pytorch](https://github.com/fanq15/caffe_to_torch_to_pytorch)
12. [pytorch-extension](https://github.com/sniklaus/pytorch-extension)：这是PyTorch的CUDA扩展，它计算两个张量的Hadamard积。
13. [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch)：该模块以张量板格式保存PyTorch张量以供检查。目前支持张量板中的标量，图像，音频，直方图功能。
14. [gpytorch](https://github.com/jrg365/gpytorch)：GPyTorch是一个高斯过程库，使用PyTorch实现。它专为轻松创建灵活的模块化高斯过程模型而设计，因此您无需成为使用GP的专家。
15. [聚焦](https://github.com/maciejkula/spotlight)：使用PyTorch的深度推荐模型。
16. [pytorch-cns](https://github.com/awentzonline/pytorch-cns)：使用PyTorch进行压缩网​​络搜索
17. [pyinn](https://github.com/szagoruyko/pyinn)：CuPy融合PyTorch神经网络运算
18. [inferno](https://github.com/nasimrahaman/inferno)：PyTorch周围的实用程序库
19. [pytorch-fitmodule](https://github.com/henryre/pytorch-fitmodule)：PyTorch模块的超简单拟合方法
20. [inferno-sklearn](https://github.com/dnouri/inferno)：一个[包含pytorch](https://github.com/dnouri/inferno)的scikit-learn兼容神经网络库。
21. [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert)：在pytorch，caffe prototxt / weights和darknet cfg / weights之间[进行转换](https://github.com/marvis/pytorch-caffe-darknet-convert)
22. [pytorch2caffe](https://github.com/longcw/pytorch2caffe)：将PyTorch模型转换为Caffemodel
23. [pytorch-tools](https://github.com/nearai/pytorch-tools)：[PyTorch的工具](https://github.com/nearai/pytorch-tools)
24. [sru](https://github.com/taolei87/sru)：训练[RNN](https://github.com/taolei87/sru)与CNN一样快（arxiv.org/abs/1709.02755）
25. [torch2coreml](https://github.com/prisma-ai/torch2coreml)：Torch7 - > CoreML
26. [PyTorch编码](https://github.com/zhanghang1989/PyTorch-Encoding)：PyTorch深度纹理编码网络[http://hangzh.com/PyTorch-Encoding](http://hangzh.com/PyTorch-Encoding)
27. [pytorch-ctc](https://github.com/ryanleary/pytorch-ctc)：PyTorch-CTC是PyTorch的CTC（连接主义时间分类）波束搜索解码的实现。C ++代码从TensorFlow中大量借用，并进行了一些改进以增加灵活性。
28. [candlegp](https://github.com/t-vi/candlegp)：[Pytorch中的](https://github.com/t-vi/candlegp)高斯过程。
29. [dpwa](https://github.com/loudinthecloud/dpwa)：通过Pair-Wise Averaging进行分布式学习。
30. [dni-pytorch](https://github.com/koz4k/dni-pytorch)：使用PyTorch的合成梯度解耦神经接口。
31. [skorch](https://github.com/dnouri/skorch)：[包含pytorch](https://github.com/dnouri/skorch)的scikit-learn兼容神经网络库
32. [点燃](https://github.com/pytorch/ignite)：Ignite是一个高级库，可以帮助在PyTorch中训练神经网络。
33. [阿诺德](https://github.com/glample/Arnold)：阿诺德 \- DOOM特工
34. [pytorch-mcn](https://github.com/albanie/pytorch-mcn)：将模型从MatConvNet转换为PyTorch
35. [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)：简化R-CNN的简化实施，具有竞争力的性能。
36. [generative_zoo](https://github.com/DL-IT/generative_zoo)：generative_zoo是一个存储库，提供PyTorch中一些生成模型的工作实现。
37. [pytorchviz](https://github.com/szagoruyko/pytorchviz)：一个小包，用于创建PyTorch执行图的可视化。
38. [cogitare](https://github.com/cogitare-ai/cogitare)：Cogitare - Python中的一个现代，快速，模块化的深度学习和机器学习框架。
39. [pydlt](https://github.com/dmarnerides/pydlt)：基于PyTorch的深度学习工具箱
40. [semi-supervised-pytorch](https://github.com/wohlert/semi-supervised-pytorch)：在[PyTorch中](https://github.com/wohlert/semi-supervised-pytorch)实现不同的基于VAE的半监督和生成模型。
41. [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster)：优化图集群算法的PyTorch扩展库。
42. [neural-assembly-compiler](https://github.com/aditya-khant/neural-assembly-compiler)：基于自适应神经编译的pyTorch神经汇编编译器。
43. [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch)：将Caffe模型转换为PyTorch。
44. [extension-cpp](https://github.com/pytorch/extension-cpp)：PyTorch中的C ++扩展
45. [pytoune](https://github.com/GRAAL-Research/pytoune)：[PyTorch](https://github.com/GRAAL-Research/pytoune)的类似Keras的框架和实用程序
46. [jetson-reinforcement](https://github.com/dusty-nv/jetson-reinforcement)：用于NVIDIA Jetson TX1 / TX2的深度强化学习库，包括PyTorch，OpenAI Gym和Gazebo robotics模拟器。
47. [matchbox](https://github.com/salesforce/matchbox)：在各个示例的级别编写PyTorch代码，然后在minibatches上高效运行。
48. [torch-two-sample](https://github.com/josipd/torch-two-sample)：用于双样本测试的PyTorch库
49. [pytorch-摘要](https://github.com/sksq96/pytorch-summary)：在PyTorch模型汇总类似于`model.summary()`在Keras
50. [mpl.pytorch](https://github.com/BelBES/mpl.pytorch)：MaxPoolingLoss的Pytorch实现。
51. [scVI-dev](https://github.com/YosefLab/scVI-dev)：PyTorch中scVI项目的开发分支
52. [apex](https://github.com/NVIDIA/apex)：实验性PyTorch扩展（稍后将弃用）
53. [ELF](https://github.com/pytorch/ELF)：ELF：游戏研究的平台。
54. [Torchlite](https://github.com/EKami/Torchlite)：一个高级图书馆（不仅仅是）Pytorch
55. [joint-vae](https://github.com/Schlumberger/joint-vae)：Pytorch实施的JointVAE，一个解开连续和离散变异因子star2的框架
56. [SLM-Lab](https://github.com/kengz/SLM-Lab)：PyTorch中的模块化深度强化学习框架。
57. [bindsnet](https://github.com/Hananel-Hazan/bindsnet)：一个Python包，用于使用PyTorch在CPU或GPU上模拟尖峰神经网络（SNN）
58. [pro\_gan\_pytorch](https://github.com/akanimax/pro_gan_pytorch)：作为PyTorch nn.Module的扩展实现的ProGAN包
59. [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)：[PyTorch的](https://github.com/rusty1s/pytorch_geometric)几何深度学习扩展库
60. [torchplus](https://github.com/knighton/torchplus)：在PyTorch模块上实现+运算符，返回序列。
61. [lagom](https://github.com/zuoxingdong/lagom)：lagom：一种轻型PyTorch基础设施，可快速构建强化学习算法原型。
62. [火炬手](https://github.com/ecs-vlc/torchbearer)：火炬手：使用PyTorch的研究人员的模型训练库。
63. [pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl)：在Pytorch中使用模型不可知元学习进行强化学习。
64. [NALU](https://github.com/bharathgs/NALU)：来自神经算术逻辑单元的NAC / NALU的基本pytorch实现，作者：trask et.al arxiv.org/pdf/1808.00508.pdf
65. [QuCumber](https://github.com/PIQuIL/QuCumber)：神经网络多体波函数重建
66. [磁铁](https://github.com/MagNet-DL/magnet)：自学的深度学习项目[http://magnet-dl.readthedocs.io/](http://magnet-dl.readthedocs.io/)
67. [opencv_transforms](https://github.com/jbohnslav/opencv_transforms)：OpenCV实现Torchvision的图像增强
68. [fastai](https://github.com/fastai/fastai)：fast.ai深度学习库，课程和教程
69. [pytorch-dense-correspondence](https://github.com/RobotLocomotion/pytorch-dense-correspondence)：“密集对象网络：学习密集视觉对象描述符和机器人操作的代码”arxiv.org/pdf/1806.08756.pdf
70. [colorization-pytorch](https://github.com/richzhang/colorization-pytorch)：交互式深色着色的PyTorch重新实现richzhang.github.io/ideepcolor
71. [beauty-net](https://github.com/cms-flash/beauty-net)：PyTorch的一个简单，灵活且可扩展的模板。很美丽。
72. [OpenChem](https://github.com/Mariewelt/OpenChem)：OpenChem：用于计算化学和药物设计研究的深度学习工具包mariewelt.github.io/OpenChem
73. [torchani](https://github.com/aiqm/torchani)：PyTorch aiqm.github.io/torchani上精确的神经网络潜力
74. [PyTorch-LBFGS](https://github.com/hjmshi/PyTorch-LBFGS)：L-BFGS的PyTorch实现。
75. [gpytorch](https://github.com/cornellius-gp/gpytorch)：[PyTorch](https://github.com/cornellius-gp/gpytorch)中高斯过程的高效和模块化实现。
76. [粗麻布](https://github.com/mariogeiger/hessian)：pytorch中的[粗](https://github.com/mariogeiger/hessian)麻布。
77. [vel](https://github.com/MillionIntegrals/vel)：深度学习研究中的速度。
78. [nonechucks](https://github.com/msamogh/nonechucks)：跳过PyTorch DataLoader中的坏项，使用Transforms as Filters等等！
79. [torchstat](https://github.com/Swall0w/torchstat)：[PyTorch中的](https://github.com/Swall0w/torchstat)模型分析器。
80. [QNNPACK](https://github.com/pytorch/QNNPACK)：量化神经网络包 \- 量化神经网络运营商的移动优化实现。
81. [torchdiffeq](https://github.com/rtqichen/torchdiffeq)：具有完全GPU支持和O（1）-memory反向传播的可区分ODE求解器。
82. [redner](https://github.com/BachiLi/redner)：可微分的蒙特卡罗路径示踪剂
83. [pixyz](https://github.com/masa-su/pixyz)：一个用于以更简洁，直观和可扩展的方式开发深度生成模型的库。
84. [euclidesdb](https://github.com/perone/euclidesdb)：嵌入数据库[http://euclidesdb.readthedocs.io的](http://euclidesdb.readthedocs.io/)多模型机器学习功能[](http://euclidesdb.readthedocs.io/)
85. [pytorch2keras](https://github.com/nerox8664/pytorch2keras)：将PyTorch动态图转换为Keras模型。
86. [沙拉](https://github.com/domainadaptation/salad)：半监督学习和领域适应。
87. [netharn](https://github.com/Erotemic/netharn)：[pytorch的](https://github.com/Erotemic/netharn)参数化拟合和预测线束。
88. [dgl](https://github.com/dmlc/dgl)：Python包，用于在现有DL框架之上轻松深入学习图形。[http://dgl.ai](http://dgl.ai/)。
89. [gandissect](https://github.com/CSAILVision/gandissect)：基于Pytorch的工具，用于可视化和理解GAN的神经元。gandissect.csail.mit.edu
90. [delira](https://github.com/justusschock/delira)：用于快速原型[制作的](https://github.com/justusschock/delira)轻量级框架，用于医学成像中的深度神经网络delira.rtfd.io
91. [蘑菇](https://github.com/AIRLab-POLIMI/mushroom)：用于强化学习实验的Python库。
92. [Xlearn](https://github.com/thuml/Xlearn)：转学习图书馆
93. [geoopt](https://github.com/ferrine/geoopt)：具有pytorch optim的黎曼自适应优化方法
94. [素食主义者](https://github.com/unit8co/vegans)：一个在PyTorch中提供各种现有GAN的图书馆。
95. [火炬测量](https://github.com/arraiyopensource/torchgeometry)：TGM：PyTorch几何
96. [AdverTorch](https://github.com/BorealisAI/advertorch)：用于对抗鲁棒性（攻击/防御/训练）研究的工具箱
97. [AdaBound](https://github.com/Luolc/AdaBound)：一个优化器，可以像Adam一样快速训练，也可以像SGD.a一样快速训练
98. [fenchel-young-loss](https://github.com/mblondel/fenchel-young-losses)：PyTorch中的概率分类/ TensorFlow / scikit-学习Fenchel-Young损失
99. [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)：计算PyTorch模型的FLOP。
100. [Tor10](https://github.com/kaihsin/Tor10)：一个通用的Tensor网络库，专为量子模拟而设计，基于pytorch。
101. [Catalyst](https://github.com/catalyst-team/catalyst)：用于PyTorch DL和RL研究的高级工具。它的开发重点是重现性，快速实验和重用代码/思想。能够研究/开发新的东西，而不是写另一个常规的火车循环。
102. [Ax](https://github.com/facebook/Ax)：自适应实验平台
103. [pywick](https://github.com/achaiah/pywick)：用于Pytorch的高级电池包含的神经网络训练库
104. [torchgpipe](https://github.com/kakaobrain/torchgpipe)：PyTorch中的GPipe实现torchgpipe.readthedocs.io
105. [hub](https://github.com/pytorch/hub)：Pytorch Hub是一个经过预先培训的模型库，旨在促进研究的可重复性。
106. [pytorch-lightning](https://github.com/williamFalcon/pytorch-lightning)：[Pytorch的](https://github.com/williamFalcon/pytorch-lightning)快速研究框架。研究员的keras版本。
107. [Tor10](https://github.com/kaihsin/Tor10)：一个通用的Tensor网络库，专为量子模拟而设计，基于pytorch。
108. [tensorwatch](https://github.com/microsoft/tensorwatch)：Microsoft Research的深度学习和强化学习的调试，监控和可视化。

## [](https://github.com/qbmzc/Awesome-pytorch-list#tutorials--examples)教程和示例

1. **[实用的Pytorch](https://github.com/spro/practical-pytorch)**：解释不同RNN模型的教程
2. [DeepLearningForNLPInPytorch](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)：关于深度学习的IPython Notebook教程，重点是自然语言处理。
3. [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)：研究人员用pytorch学习深度学习的教程。
4. [pytorch-exercises](https://github.com/keon/pytorch-exercises)：pytorch-exercises集合。
5. [pytorch教程](https://github.com/pytorch/tutorials)：各种pytorch教程。
6. [pytorch examples](https://github.com/pytorch/examples)：展示使用pytorch的示例的存储库
7. [pytorch practice](https://github.com/napsternxg/pytorch-practice)：[pytorch上的](https://github.com/napsternxg/pytorch-practice)一些示例脚本。
8. [pytorch迷你教程](https://github.com/vinhkhuc/PyTorch-Mini-Tutorials)：[PyTorch的](https://github.com/vinhkhuc/PyTorch-Mini-Tutorials)最小教程改编自Alec Radford的Theano教程。
9. [pytorch文本分类](https://github.com/xiayandi/Pytorch_text_classification)：[Pytorch中](https://github.com/xiayandi/Pytorch_text_classification)基于CNN的文本分类的简单实现
10. [猫与狗](https://github.com/desimone/pytorch-cat-vs-dogs)：在pytorch中进行网络微调的例子，对于kaggle比赛Dogs vs. Cats Redux：Kernels Edition。目前在排行榜上排名第27（0.05074）。
11. [convnet](https://github.com/eladhoffer/convNet.pytorch)：这是深度卷积网络在各种数据集（ImageNet，Cifar10，Cifar100，MNIST）上的完整培训示例。
12. [pytorch-generative-adversarial-networks](https://github.com/mailmahee/pytorch-generative-adversarial-networks)：使用PyTorch的简单生成对抗网络（GAN）。
13. [pytorch容器](https://github.com/amdegroot/pytorch-containers)：该存储库旨在通过提供Torch表层的PyTorch实现列表，帮助前Torchies更加无缝地过渡到PyTorch的“无容器”世界。
14. [pytorch中的T-SNE](https://github.com/cemoody/topicsne)：[pytorch中的](https://github.com/cemoody/topicsne) t-SNE实验
15. [AAE_pytorch](https://github.com/fducau/AAE_pytorch)：Adversarial Autoencoders（与Pytorch合作）。
16. [Kind\_PyTorch\_Tutorial](https://github.com/GunhoChoi/Kind_PyTorch_Tutorial)：适合初学者的PyTorch教程。
17. [pytorch-poetry-gen](https://github.com/justdark/pytorch-poetry-gen)：基于pytorch的char-RNN。
18. [pytorch-REINFORCE](https://github.com/JamesChuanggg/pytorch-REINFORCE)：REINFORCE的PyTorch实现，这个repo支持OpenAI gym中的连续和离散环境。
19. **[PyTorch-Tutorial](https://github.com/MorvanZhou/PyTorch-Tutorial)**：轻松快速地构建您的神经网络 [https://morvanzhou.github.io/tutorials/](https://morvanzhou.github.io/tutorials/)
20. [pytorch-intro](https://github.com/joansj/pytorch-intro)：一些脚本，用于说明如何在PyTorch中执行CNN和RNN
21. [pytorch-classification](https://github.com/bearpaw/pytorch-classification)：CIFAR-10/100和ImageNet上图像分类任务的统一框架。
22. [pytorch_notebooks - hardmaru](https://github.com/hardmaru/pytorch_notebooks)：在NumPy和PyTorch中创建的随机教程。
23. [pytorch_tutoria-quick](https://github.com/soravux/pytorch_tutorial)：快速PyTorch介绍和教程。目标是计算机视觉，图形和机器学习研究人员渴望尝试新的框架。
24. [Pytorch\_fine\_tuning_Tutorial](https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial)：关于在PyTorch中执行微调或转移学习的简短教程。
25. [pytorch_exercises](https://github.com/Kyubyong/pytorch_exercises)：pytorch-exercises
26. [交通标志检测](https://github.com/soumith/traffic-sign-detection-homework)：nyu-cv-fall-2017示例
27. [mss_pytorch](https://github.com/Js-Mim/mss_pytorch)：通过循环推理和跳过过滤连接进行语音分离 \- PyTorch实现。演示：js-mim.github.io/mss_pytorch
28. [DeepNLP-models-Pytorch](https://github.com/DSKSD/DeepNLP-models-Pytorch) Pytorch在cs-224n中实现各种Deep NLP模型（Stanford Univ：NLP with Deep Learning）
29. [Mila入门教程](https://github.com/mila-udem/welcome_tutorials)：为欢迎MILA的新生提供各种教程。
30. [pytorch.rl.learning](https://github.com/moskomule/pytorch.rl.learning)：用于使用PyTorch学习强化学习。
31. [minimal-seq2seq](https://github.com/keon/seq2seq)：在PyTorch中注意神经机器翻译的最小Seq2Seq模型
32. [tensorly-notebooks](https://github.com/JeanKossaifi/tensorly-notebooks)：[TensorLy中的](https://github.com/JeanKossaifi/tensorly-notebooks) Tensor方法tensorly.github.io/dev
33. [pytorch_bits](https://github.com/jpeg729/pytorch_bits)：时间序列预测相关的例子。
34. [skip-thoughts](https://github.com/sanyam5/skip-thoughts)：在PyTorch中实现Skip-Thought Vectors。
35. [video-caption-pytorch](https://github.com/xiadingZ/video-caption-pytorch)：用于视频字幕的pytorch代码。
36. [Capsule-Network-Tutorial](https://github.com/higgsfield/Capsule-Network-Tutorial)：Pytorch易于关注的胶囊网络教程。
37. [代码学习深度学习与pytorch](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch)：这是“使用PyTorch学习深度学习”一书的代码item.jd.com/17915495606.html
38. [RL-Adventure](https://github.com/higgsfield/RL-Adventure)：Pytorch易于遵循的深度Q学习教程，带有清晰的可读代码。
39. [accelerated\_dl\_pytorch](https://github.com/hpcgarage/accelerated_dl_pytorch)：在亚特兰大二世的Jupyter Day与PyTorch加速深度学习。
40. [RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2)：PyTorch4教程：演员评论家/近端政策优化/宏碁/ ddpg /双人决斗ddpg /软演员评论家/生成对抗模仿学习/后见之明体验重播
41. [50行代码中的生成对抗网络（GAN）（PyTorch）](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f)
42. [对抗性\-自动编码与 \- pytorch](https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/)
43. [使用pytorch转移学习](https://medium.com/@vishnuvig/transfer-learning-using-pytorch-4c3475f4495)
44. [如何对实施-A-YOLO对象检测器功能于pytorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
45. [pytorch换推荐器-101](http://blog.fastforwardlabs.com/2018/04/10/pytorch-for-recommenders-101.html)
46. [pytorch换numpy的用户](https://github.com/wkentaro/pytorch-for-numpy-users)
47. [PyTorch教程](http://www.pytorchtutorial.com/)：PyTorch中文教程。
48. [grokking-pytorch](https://github.com/Kaixhin/grokking-pytorch)：Hitchiker的PyTorch指南
49. [PyTorch-Deep-Learning-Minicourse](https://github.com/Atcold/PyTorch-Deep-Learning-Minicourse)：使用PyTorch进行深度学习的微创。
50. [pytorch-custom-dataset-examples](https://github.com/utkuozbulak/pytorch-custom-dataset-examples)：PyTorch的一些自定义数据集示例
51. [基于序列的推荐者的乘法LSTM](https://florianwilhelm.info/2018/08/multiplicative_LSTM_for_sequence_based_recos/)
52. [deeplearning.ai-pytorch](https://github.com/furkanu/deeplearning.ai-pytorch)：课程深度学习（deeplearning.ai）专业化的PyTorch实现。
53. [MNIST\_Pytorch\_python\_and\_capi](https://github.com/tobiascz/MNIST_Pytorch_python_and_capi)：这是一个如何在Python中训练MNIST网络并使用pytorch 1.0在c ++中运行它的示例
54. [torch_light](https://github.com/ne7ermore/torch_light)：教程和示例包括强化训练，NLP，CV
55. [portrain-gan](https://github.com/dribnet/portrain-gan)：用于解码（并几乎编码）来自art-DCGAN的Portrait GAN的潜伏的火炬代码。
56. [mri-analysis-pytorch](https://github.com/omarsar/mri-analysis-pytorch)：使用PyTorch和MedicalTorch进行MRI分析
57. [cifar10-fast](https://github.com/davidcpage/cifar10-fast)：如本[博客系列](https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/)所述，在79秒内演示如何在CIFAR10上训练小型ResNet至94％的测试精度。
58. [PyTorch深度学习简介](https://in.udacity.com/course/deep-learning-pytorch--ud188)：Udacity和facebook的免费课程，PyTorch的精彩介绍，以及对PyTorch原创作者之一Soumith Chintala的采访。
59. [pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)：用于开始使用PyTorch和TorchText进行情绪分析的教程。
60. [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)：PyTorch图像模型，脚本，预训练权重 - （SE）ResNet / ResNeXT，DPN，EfficientNet，MobileNet-V3 / V2 / V1，MNASNet，单路径NAS，FBNet等。

## [](https://github.com/qbmzc/Awesome-pytorch-list#paper-implementations)论文实施

1. [google_evolution](https://github.com/neuralix/google_evolution)：这实现了Esteban Real等人大规模演化图像分类器的结果网络之一。人。
2. [pyscatwave](https://github.com/edouardoyallon/pyscatwave)：使用CuPy / PyTorch进行快速散射变换，请阅读[此处](https://arxiv.org/abs/1703.08961)的论文[](https://arxiv.org/abs/1703.08961)
3. [scalingscattering](https://github.com/edouardoyallon/scalingscattering)：扩展散射变换：深度混合网络。
4. [深度自动标点符号](https://github.com/episodeyang/deep-auto-punctuation)：按字符逐个学习的自动标点符号的pytorch实现。
5. [Realtime\_Multi-Person\_Pose_Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)：这是[Realtime\_Multi-Person\_Pose_Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)的pytorch版本，原始代码在[这里](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)。
6. [PyTorch-value-iteration-networks](https://github.com/onlytailei/PyTorch-value-iteration-networks)：价值迭代网络（NIPS '16）论文的PyTorch实现
7. [pytorch_Highway](https://github.com/analvikingur/pytorch_Highway)：在pytorch中实施的公路网。
8. [pytorch\_NEG\_loss](https://github.com/analvikingur/pytorch_NEG_loss)：在pytorch中实现的NEG丢失。
9. [pytorch_RVAE](https://github.com/analvikingur/pytorch_RVAE)：循环变分自动编码器，用于生成在pytorch中实现的顺序数据。
10. [pytorch_TDNN](https://github.com/analvikingur/pytorch_TDNN)：在pytorch中实现的时间延迟NN。
11. [eve.pytorch](https://github.com/moskomule/eve.pytorch)：Eve Optimizer的一个实现，在具有反馈的Imploving随机梯度下降中提出，Koushik和Hayashi，2016。
12. [e2e-model-learning](https://github.com/locuslab/e2e-model-learning)：基于任务的端到端模型学习。
13. [pix2pix-pytorch](https://github.com/mrzhu-cool/pix2pix-pytorch)：PyTorch实现的“使用条件对抗网络的图像到图像的翻译”。
14. [单次多](https://github.com/amdegroot/ssd.pytorch)盒[检测器](https://github.com/amdegroot/ssd.pytorch)：单次多盒[检测器](https://github.com/amdegroot/ssd.pytorch)的PyTorch实现。
15. [DiscoGAN](https://github.com/carpedm20/DiscoGAN-pytorch)：PyTorch实施“学习发现与生成对抗网络的跨域关系”
16. [官方DiscoGAN实施](https://github.com/SKTBrain/DiscoGAN)：“学习发现与生成性对抗网络的跨域关系”的官方实施。
17. [pytorch-es](https://github.com/atgambardella/pytorch-es)：这是[Evolution Strategies](https://arxiv.org/abs/1703.03864)的PyTorch实现。
18. [piwise](https://github.com/bodokaiser/piwise)：使用pytorch对VOC2012数据集进行像素分割。
19. [pytorch-dqn](https://github.com/transedward/pytorch-dqn)：[pytorch中的](https://github.com/transedward/pytorch-dqn)深度Q-Learning网络。
20. [neuraltalk2-pytorch](https://github.com/ruotianluo/neuraltalk2.pytorch)：[pytorch中的](https://github.com/ruotianluo/neuraltalk2.pytorch)图像字幕模型（with_finetune分支中的finetunable cnn）
21. [vnet.pytorch](https://github.com/mattmacy/vnet.pytorch)：V-Net的Pytorch实现：用于体积医学图像分割的完全卷积神经网络。
22. [pytorch-fcn](https://github.com/wkentaro/pytorch-fcn)：完全卷积网络的PyTorch实现。
23. [WideResNets](https://github.com/xternalz/WideResNet-pytorch)：PyTorch中实现的CIFAR10 / 100的WideResNets。这种实现需要的GPU内存少于官方Torch实现所需的内存：[https](https://github.com/szagoruyko/wide-residual-networks)：[//github.com/szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)。
24. [pytorch\_highway\_networks](https://github.com/c0nn3r/pytorch_highway_networks)：在PyTorch中实现的高速公路网络。
25. [pytorch-NeuCom](https://github.com/ypxie/pytorch-NeuCom)：Pytorch实现了DeepMind的可微分神经计算机论文。
26. [captionGen](https://github.com/eladhoffer/captionGen)：使用PyTorch为图像生成标题。
27. [AnimeGAN](https://github.com/jayleicn/animeGAN)：生成对抗网络的简单PyTorch实现，专注于动漫人脸绘图。
28. [Cnn-text分类](https://github.com/Shawn1993/cnn-text-classification-pytorch)：这是PyTorch中用于句子分类论文的Kim's卷积神经网络的实现。
29. [deepspeech2](https://github.com/SeanNaren/deepspeech.pytorch)：使用百度Warp-CTC实现DeepSpeech2。创建基于DeepSpeech2架构的网络，该架构使用CTC激活功能进行培训。
30. [seq2seq](https://github.com/MaximumEntropy/Seq2Seq-PyTorch)：此存储库包含[PyTorch](https://github.com/MaximumEntropy/Seq2Seq-PyTorch)中的Sequence to Sequence（Seq2Seq）模型的实现
31. [PyTorch中的Asynchronous Advantage Actor-Critic](https://github.com/rarilurelo/pytorch_a3c)：这是针对深度强化学习的异步方法中描述的A3C的PyTorch实现。由于PyTorch有一个简单的方法来控制多进程内的共享内存，我们可以轻松实现像A3C这样的异步方法。
32. [densenet](https://github.com/bamos/densenet.pytorch)：这是DenseNet-BC架构的PyTorch实现，如G. Huang，Z。Liu，K。Weinberger和L. van der Maaten所着的Pensely Connected Convolutional Networks一文中所述。这个实现的CIFAR-10 +错误率为4.77，其中100层DenseNet-BC的增长率为12.他们的官方实现和许多其他第三方实现的链接可以在GitHub上的liuzhuang13 / DenseNet repo中找到。
33. [nninit](https://github.com/alykhantejani/nninit)：PyTorch nn.Modules的权重初始化方案。这是@kaixhin对Torch7的流行nninit的一个端口。
34. [更快的rcnn](https://github.com/longcw/faster_rcnn_pytorch)：这是[更快的RCNN](https://github.com/longcw/faster_rcnn_pytorch)的PyTorch实现。该项目主要基于py-faster-rcnn和TFFRCNN。有关R-CNN的详细信息，请参阅文章更快的R-CNN：通过区域提案网络实现实时目标检测由邵少卿，何开明，Ross Girshick，孙健
35. [doomnet](https://github.com/akolishchak/doom-net-pytorch)：PyTorch的Doom-net版本在ViZDoom环境中实现了一些RL模型。
36. [flownet](https://github.com/ClementPinard/FlowNetPytorch)：Dosovitskiy等人的Pytorch实施FlowNet。
37. [sqeezenet](https://github.com/gsp-27/pytorch_Squeezenet)：在pytorch中实现Squeezenet，在CIFAR10数据上实现####预训练模型计划在cifar 10上训练模型并添加块连接。
38. [WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN)：在pytorch中的wassersteinGAN。
39. [optnet](https://github.com/locuslab/optnet)：此存储库由Brandon Amos和J. Zico Kolter提供，包含PyTorch源代码，用于在我们的论文OptNet中重现实验：可区分优化作为神经网络中的一层。
40. [qp求解器](https://github.com/locuslab/qpth)：PyTorch的快速且可微分的QP求解器。由Brandon Amos和J. Zico Kolter精心打造。
41. [基于模型的加速](https://github.com/ikostrikov/pytorch-naf)连续深度Q学习：[基于模型的加速](https://github.com/ikostrikov/pytorch-naf)实现连续深度Q学习。
42. [学习通过梯度下降的梯度下降](https://github.com/ikostrikov/pytorch-meta-optimizer)来学习：PyTorch实现学习通过梯度下降的梯度下降来学习。
43. [快速神经风格](https://github.com/darkstar112358/fast-neural-style)：[快速神经风格的](https://github.com/darkstar112358/fast-neural-style) pytorch实现，该模型使用[实时样式转移和超分辨率的感知损失中](https://arxiv.org/abs/1603.08155)描述的方法以及实例规范化。
44. [PytorchNeuralStyleTransfer](https://github.com/leongatys/PytorchNeuralStyleTransfer)：在Pytorch中实现神经样式传递。
45. [Pytorch图像风格变换的](https://github.com/bengxy/FastNeuralStyle)快速神经风格：[Pytorch图像风格变换的](https://github.com/bengxy/FastNeuralStyle)快速神经风格。
46. [神经风格转移](https://github.com/alexis-jacq/Pytorch-Tutorials)：通过Leon A. Gatys，Alexander S. Ecker和Matthias Bethge开发的神经风格算法（[https://arxiv.org/abs/1508.06576](https://arxiv.org/abs/1508.06576)）介绍PyTorch。
47. [VIN\_PyTorch\_Visdom](https://github.com/zuoxingdong/VIN_PyTorch_Visdom)：Value迭代网络（VIN）的PyTorch实现：清洁，简单和模块化。Visdom中的可视化。
48. [YOLO2](https://github.com/longcw/yolo2-pytorch)：PyTorch中的YOLOv2。
49. [注意力转移](https://github.com/szagoruyko/attention-transfer)：pytorch中的注意力转移，请阅读[此处](https://arxiv.org/abs/1612.03928)的论文。
50. [SVHNClassifier](https://github.com/potterhsu/SVHNClassifier-PyTorch)：[使用深度卷积神经网络从街景图像中进行多位数识别的](https://arxiv.org/pdf/1312.6082.pdf) PyTorch实现。
51. [pytorch-deform-conv](https://github.com/oeway/pytorch-deform-conv)：可变形卷积的PyTorch实现。
52. [BEGAN-pytorch](https://github.com/carpedm20/BEGAN-pytorch)：PyTorch实现[BEGAN](https://arxiv.org/abs/1703.10717)：边界平衡生成对抗网络。
53. [treelstm.pytorch](https://github.com/dasguptar/treelstm.pytorch)：[PyTorch中的](https://github.com/dasguptar/treelstm.pytorch)树LSTM实现。
54. [AGE](https://github.com/DmitryUlyanov/AGE)：Dmitry Ulyanov，Andrea Vedaldi和Victor Lempitsky撰写的“Adversarial Generator-Encoder Networks”论文代码，可以在[这里](http://sites.skoltech.ru/app/data/uploads/sites/25/2017/04/AGE.pdf)找到[](http://sites.skoltech.ru/app/data/uploads/sites/25/2017/04/AGE.pdf)
55. [ResNeXt.pytorch](https://github.com/prlz77/ResNeXt.pytorch)：使用[pytorch](https://github.com/prlz77/ResNeXt.pytorch)重现ResNet-V3（深度神经网络的聚合残差变换）。
56. [pytorch-rl](https://github.com/jingweiz/pytorch-rl)：使用pytorch和visdom深度强化学习
57. [Deep-Leafsnap](https://github.com/sujithv28/Deep-Leafsnap)：与传统的计算机视觉方法相比，使用深度神经网络复制LeafSnap来测试准确性。
58. [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)：用于未配对和成对图像到图像转换的PyTorch实现。
59. [A3C-PyTorch](https://github.com/onlytailei/A3C-PyTorch)：在PyTorch中PyTorch实现Advantage异步演员评论算法（A3C）
60. [pytorch-value-iteration-networks](https://github.com/kentsommer/pytorch-value-iteration-networks)：Pytorch实现价值迭代网络（NIPS 2016年最佳论文）
61. [PyTorch-Style-Transfer](https://github.com/zhanghang1989/PyTorch-Style-Transfer)：用于实时传输的多样式生成网络的PyTorch实现
62. [pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet)：pytorch-deeplab-resnet-model。
63. [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)：“PointNet：用于3D分类和分割的点集的深度学习”的pytorch实现[https://arxiv.org/abs/1612.00593](https://arxiv.org/abs/1612.00593)
64. **[pytorch-playground](https://github.com/aaron-xichen/pytorch-playground)：[pytorch中的](https://github.com/aaron-xichen/pytorch-playground)基础预训练模型和数据集（MNIST，SVHN，CIFAR10，CIFAR100，STL10，AlexNet，VGG16，VGG19，ResNet，Inception，SqueezeNet）**。
65. [pytorch-dnc](https://github.com/jingweiz/pytorch-dnc)：神经图灵机（NTM）和可分辨神经计算机（DNC）与pytorch＆visdom。
66. [pytorch\_image\_classifier](https://github.com/jinfagang/pytorch_image_classifier)：使用Pytorch的最小但实用的图像分类器Pipline，ResNet18上的Finetune，在自己的小数据集上获得99％的准确度。
67. [mnist-svhn-transfer](https://github.com/yunjey/mnist-svhn-transfer)：用于域转移（最小）的PyTorch实现CycleGAN和SGAN。
68. [pytorch-yolo2](https://github.com/marvis/pytorch-yolo2)：pytorch-yolo2
69. [dni](https://github.com/andrewliao11/dni.pytorch)：在Pytorch中使用合成梯度实现解耦神经接口
70. [wgan-gp](https://github.com/caogang/wgan-gp)：文章“改进Wasserstein GAN训练”的实施。
71. [pytorch-seq2seq-intent-parsing](https://github.com/spro/pytorch-seq2seq-intent-parsing)：使用seq2seq +注意在PyTorch中进行Intent解析和插槽填充
72. [pyTorch_NCE](https://github.com/demelin/pyTorch_NCE)：[pyTorch](https://github.com/demelin/pyTorch_NCE)的噪声对比度估计算法的实现。工作，但效率不高。
73. [molencoder](https://github.com/cxhernandez/molencoder)：PyTorch中的分子自动编码器
74. [GAN-weight-norm](https://github.com/stormraiser/GAN-weight-norm)：“关于生成对抗网络中批量和权重标准化的影响”的代码
75. [lgamma](https://github.com/rachtsingh/lgamma)：PyTorch的polygamma，lgamma和beta函数的实现
76. [bigBatch](https://github.com/eladhoffer/bigBatch)：用于生成结果的代码“更长时间训练，更好地概括：缩小神经网络大批量训练中的泛化差距”
77. [rl\_a3c\_pytorch](https://github.com/dgriff777/rl_a3c_pytorch)：为Atari 2600实施A3C LSTM的强化学习。
78. [pytorch-retraining](https://github.com/ahirner/pytorch-retraining)：为PyTorch的模型动物园（torchvision）转移学习枪战
79. [nmp_qc](https://github.com/priba/nmp_qc)：计算机视觉的神经消息传递
80. [grad-cam](https://github.com/jacobgil/pytorch-grad-cam)：Grator-CAM的Pytorch实现
81. [pytorch-trpo](https://github.com/mjacar/pytorch-trpo)：PyTorch实施信任区域政策优化（TRPO）
82. [pytorch-explain-black-box](https://github.com/jacobgil/pytorch-explain-black-box)：PyTorch通过有意义的扰动实现[黑盒子的](https://github.com/jacobgil/pytorch-explain-black-box)可解释解释
83. [vae_vpflows](https://github.com/jmtomczak/vae_vpflows)：PyTorch中用于凸组合线性IAF和Householder Flow的代码，JM Tomczak和M. Welling [https://jmtomczak.github.io/deebmed.html](https://jmtomczak.github.io/deebmed.html)
84. [关系网络](https://github.com/kimhc6028/relational-networks)：Pytorch实现“关系推理的简单神经网络模块”（关系网络）[https://arxiv.org/pdf/1706.01427.pdf](https://arxiv.org/pdf/1706.01427.pdf)
85. [vqa.pytorch](https://github.com/Cadene/vqa.pytorch)：[Pytorch中的](https://github.com/Cadene/vqa.pytorch)视觉问题回答
86. [端到端谈判代表](https://github.com/facebookresearch/end-to-end-negotiator)：交易还是不交易？谈判对话的端到端学习
87. [odin-pytorch](https://github.com/ShiyuLiang/odin-pytorch)：神经网络中的分布式实例的原理检测。
88. [FreezeOut](https://github.com/ajbrock/FreezeOut)：通过逐步冻结层来加速神经网络训练。
89. [ARAE](https://github.com/jakezhaojb/ARAE)：Zhao，Kim，Zhang，Rush和LeCun [撰写](https://github.com/jakezhaojb/ARAE)的“用于生成离散结构的[异常](https://github.com/jakezhaojb/ARAE)正则化自动编码器”的代码。
90. [forward-thinking-pytorch](https://github.com/kimhc6028/forward-thinking-pytorch)：Pytorch实施的“前瞻思维：一次构建和训练神经网络一层” [https://arxiv.org/pdf/1706.02480.pdf](https://arxiv.org/pdf/1706.02480.pdf)
91. [context\_encoder\_pytorch](https://github.com/BoyuanJiang/context_encoder_pytorch)：PyTorch实现上下文编码器
92. [注意力是你所需要的 \- pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)：变形金刚模型中的PyTorch实现在“注意就是你所需要的一切”中。[https://github.com/thnkim/OpenFacePytorch](https://github.com/thnkim/OpenFacePytorch)
93. [OpenFacePytorch](https://github.com/thnkim/OpenFacePytorch)：使用OpenFace的nn4.small2.v1.t7模型的PyTorch模块
94. [神经组合-rl-pytorch](https://github.com/pemami4911/neural-combinatorial-rl-pytorch)：具有强化学习的神经组合优化的PyTorch实现。
95. [pytorch-nec](https://github.com/mjacar/pytorch-nec)：神经情景控制（NEC）的PyTorch实现
96. [seq2seq.pytorch](https://github.com/eladhoffer/seq2seq.pytorch)：使用PyTorch进行序列到序列学习
97. [Pytorch-Sketch-RNN](https://github.com/alexis-jacq/Pytorch-Sketch-RNN)：arxiv.org/abs/1704.03477的pytorch实现
98. [pytorch-pruning](https://github.com/jacobgil/pytorch-pruning)：PyTorch实现\[1611.06440\]修剪卷积神经网络进行资源有效推理
99. [DrQA](https://github.com/hitvoice/DrQA)：阅读维基百科以解答开放域问题的pytorch实现。
100. [YellowFin_Pytorch](https://github.com/JianGoForIt/YellowFin_Pytorch)：自动调整动量SGD优化器
101. [samplernn-pytorch](https://github.com/deepsound-project/samplernn-pytorch)：SampleRNN的PyTorch实现：无条件端到端神经音频生成模型。
102. [AEGeAN](https://github.com/tymokvo/AEGeAN)：具有AE稳定性的更深入的DCGAN
103. [/ pytorch-SRResNet](https://github.com/twtygqyy/pytorch-SRResNet)：使用生成的对抗网络进行照片真实单图像超分辨率的pytorch实现arXiv：1609.04802v2
104. [vsepp](https://github.com/fartashf/vsepp)：论文的代码“VSE ++：改进的视觉语义嵌入”
105. [Pytorch-DPPO](https://github.com/alexis-jacq/Pytorch-DPPO)：分布式近端策略优化的Pytorch实现：arxiv.org/abs/1707.02286
106. [单位](https://github.com/mingyuliutw/UNIT)：PyTorch实现我们的耦合VAE-GAN算法用于无监督的图像到图像转换
107. [efficient\_densenet\_pytorch](https://github.com/gpleiss/efficient_densenet_pytorch)：DenseNets的内存高效实现
108. [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch)：[PyTorch中的](https://github.com/yjxiong/tsn-pytorch)时间段网络（TSN）。
109. [SMASH](https://github.com/ajbrock/SMASH)：一种有效探索神经架构的实验技术。
110. [pytorch-retinanet](https://github.com/kuangliu/pytorch-retinanet)：PyTorch中的RetinaNet
111. [biogans](https://github.com/aosokin/biogans)：支持ICCV 2017论文“生物图像合成的GAN”的实施。
112. [通过对抗性学习进行语义图像合成](https://github.com/woozzu/dong_iccv_2017)：在ICCV 2017中的文章“通过对抗性学习进行语义图像合成”的PyTorch实现。
113. [fmpytorch](https://github.com/jmhessel/fmpytorch)：cython中的分解机器模块的PyTorch实现。
114. [ORN](https://github.com/ZhouYanzhao/ORN)：2017年CVPR中“定向响应网络”一文的PyTorch实现。
115. [pytorch-maml](https://github.com/katerakelly/pytorch-maml)：MAML的PyTorch实现：arxiv.org/abs/1703.03400
116. [pytorch-generative-model-collections](https://github.com/znxlwm/pytorch-generative-model-collections)：Pytorch版本中生成模型的集合。
117. [vqa-winner-cvprw-2017](https://github.com/markdtw/vqa-winner-cvprw-2017)：Pytorch在CVPR'17中实施VQA Chllange研讨会的获奖者。
118. [tacotron_pytorch](https://github.com/r9y9/tacotron_pytorch)：PyTorch实现了Tacotron语音合成模型。
119. [pspnet-pytorch](https://github.com/Lextal/pspnet-pytorch)：PyTorch实现PSPNet分段网络
120. [LM-LSTM-CRF](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF)：使用任务感知语言模型增强序列标记[http://arxiv.org/abs/1709.04109](http://arxiv.org/abs/1709.04109)
121. [面部对齐](https://github.com/1adrianb/face-alignment)：Pytorch实施论文“我们在多大程度上解决了2D和3D人脸对齐问题？（以及230,000个3D面部地标的数据集）”，ICCV 2017
122. [DepthNet](https://github.com/ClementPinard/DepthNet)：关于Still Box数据集的PyTorch DepthNet培训。
123. [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch)：PyTorch版本的文章'用于单图像超分辨率的增强型深度残留网络'（CVPRW 2017）
124. [e2c-pytorch](https://github.com/ethanluoyc/e2c-pytorch)：在PyTorch中嵌入Control实现。
125. [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)：用于动作识别的3D ResNets。
126. [bandit-nmt](https://github.com/khanhptnk/bandit-nmt)：这是我们的EMNLP 2017论文“使用模拟人体反馈进行强盗神经机器翻译的强化学习”的代码回购，它在神经编码器 - 解码器模型之上实现A2C算法，并在模拟噪声奖励下对组合进行基准测试。
127. [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)：使用Kronecker因子近似（ACKTR）进行深度强化学习的Advantage Actor Critic（A2C），近端策略优化（PPO）和可扩展信任域方法的PyTorch实现。
128. [zalando-pytorch](https://github.com/baldassarreFe/zalando-pytorch)：来自Zalando 的[Fashion-MNIST](https://github.com/qbmzc/Awesome-pytorch-list/blob/master/zalandoresearch/fashion-mnist)数据集的各种实验。
129. [sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch)：SphereFace的PyTorch实现。
130. [分类DQN](https://github.com/floringogianu/categorical-dqn)：从[分布视角看强化学习](https://arxiv.org/abs/1707.06887)的分类DQN的PyTorch实现。
131. [pytorch-ntm](https://github.com/loudinthecloud/pytorch-ntm)：pytorch ntm实现。
132. [mask\_rcnn\_pytorch](https://github.com/felixgwu/mask_rcnn_pytorch)：在PyTorch中屏蔽RCNN。
133. [graph\_convnets\_pytorch](https://github.com/xbresson/graph_convnets_pytorch)：图形ConvNets，NIPS'16的PyTorch实现
134. [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)：基于Xinlei Chen的tf-faster-rcnn的更快的RCNN检测框架的pytorch实现。
135. [torchMoji](https://github.com/huggingface/torchMoji)：DeepMoji模型的pyTorch实现：用于分析情绪，情绪，讽刺等的最先进的深度学习模型。
136. [semantic-segmentation-pytorch](https://github.com/hangzhaomit/semantic-segmentation-pytorch)：[MIT ADE20K数据集](http://sceneparsing.csail.mit.edu/)上语义分割/场景解析的Pytorch实现[](http://sceneparsing.csail.mit.edu/)
137. [pytorch-qrnn](https://github.com/salesforce/pytorch-qrnn)：准回归神经网络的PyTorch实现 - 比NVIDIA的cuDNN LSTM快16倍
138. [pytorch-sgns](https://github.com/theeluwin/pytorch-sgns)：PyTorch中的Skipgram负抽样。
139. [SfmLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch)：来自Tinghui Zhou等人的Pytorch版本的SfmLearner。
140. [deformable-convolution-pytorch](https://github.com/1zb/deformable-convolution-pytorch)：[可变形卷积的PyTorch](https://github.com/1zb/deformable-convolution-pytorch)实现。
141. [skip-gram-pytorch](https://github.com/fanglanting/skip-gram-pytorch)：skipgram模型的完整pytorch实现（带子采样和负采样）。使用Spearman的秩相关性来测试嵌入结果。
142. [stackGAN-v2](https://github.com/hanzhanggit/StackGAN-v2)：用于再现StackGAN_v2的Pytorch实现结果StackGAN ++：由Han Zhang *，Tao Xu *，Hongsheng Li，Shaoting Zhang，Xiaowang Wang，Xiaolei Huang，Dimitris Metaxas组成的堆叠生成对抗网络的逼真图像合成。
143. [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch)：用于图像字幕的自我关键序列训练的非官方pytorch实现。
144. [pygcn](https://github.com/tkipf/pygcn)：[PyTorch中的](https://github.com/tkipf/pygcn)图形卷积网络。
145. [dnc](https://github.com/ixaxaar/pytorch-dnc)：可分辨的神经计算机，用于Pytorch
146. [prog\_gans\_pytorch_inference](https://github.com/ptrblck/prog_gans_pytorch_inference)：使用CelebA快照推断“GAN渐进式增长”的PyTorch。
147. [pytorch-capsule](https://github.com/timomernick/pytorch-capsule)：Pytorch实现Hinton的胶囊之间的动态路由。
148. [PyramidNet-PyTorch](https://github.com/dyhan0920/PyramidNet-PyTorch)：PyramidNets的PyTorch实现（Deep Pyramidal Residual Networks，arxiv.org [/abs](https://github.com/dyhan0920/PyramidNet-PyTorch) /1610.02915）
149. [无线电变压器网络](https://github.com/gram-ai/radio-transformer-networks)：来自“物理层深度学习简介”一文的无线电变压器网络的PyTorch实现。arxiv.org/abs/1702.00832
150. [按喇叭](https://github.com/castorini/honk)：为关键词识别谷歌的TensorFlow细胞神经网络的PyTorch重新实现。
151. [DeepCORAL](https://github.com/SSARCandy/DeepCORAL)：PyTorch实现的“Deep CORAL：Deep Domain Adaptation的相关对齐”，ECCV 2016
152. [pytorch-pose](https://github.com/bearpaw/pytorch-pose)：用于2D人体姿势估计的PyTorch工具包。
153. [lang-emerge-parlai](https://github.com/karandesai-96/lang-emerge-parlai)：使用PyTorch和ParlAI实施EMNLP 2017论文“自然语言不会在Multi-Agent对话中自然出现”
154. [彩虹](https://github.com/Kaixhin/Rainbow)：彩虹：结合深度强化学习的改进
155. [pytorch\_compact\_bilinear_pooling v1](https://github.com/gdlg/pytorch_compact_bilinear_pooling)：这个存储库有一个用于PyTorch的Compact Bilinear Pooling和Count Sketch的纯Python实现。
156. [CompactBilinearPooling-Pytorch v2](https://github.com/DeepInsight-PCALab/CompactBilinearPooling-Pytorch) :( Yang Gao，et al。）Compact Cilinear Pooling的Pytorch实现。
157. [FewShotLearning](https://github.com/gitabcworld/FewShotLearning)：Pytorch实施的论文“优化作为少数镜头学习的模型”
158. [meProp](https://github.com/jklj077/meProp)：“meProp的代码：用于加速深度学习和减少过度拟合的Sparsified Back传播”。
159. [SFD_pytorch](https://github.com/clcarwin/SFD_pytorch)：单镜头不变量人脸检测器的PyTorch实现。
160. [GradientEpisodicMemory](https://github.com/facebookresearch/GradientEpisodicMemory)：GEM的连续学习：梯度情景记忆。[https://arxiv.org/abs/1706.08840](https://arxiv.org/abs/1706.08840)
161. [DeblurGAN](https://github.com/KupynOrest/DeblurGAN)：Pytorch实施的论文DeblurGAN：使用条件对抗网络的盲运动去模糊。
162. [StarGAN](https://github.com/yunjey/StarGAN)：StarGAN：用于多域图像到图像转换的统一生成式对抗网络。
163. [CapsNet-pytorch](https://github.com/adambielski/CapsNet-pytorch)：NIPS 2017纸张胶囊间动态路由的PyTorch实现。
164. [CondenseNet](https://github.com/ShichenLiu/CondenseNet)：CondenseNet：使用学习集团卷积的高效密集网络。
165. [深度图像优先](https://github.com/DmitryUlyanov/deep-image-prior)：用神经网络恢复[图像](https://github.com/DmitryUlyanov/deep-image-prior)但没有学习。
166. [深头姿势](https://github.com/natanielruiz/deep-head-pose)：使用PyTorch进行深度学习头部姿势估计。
167. [随机擦除](https://github.com/zhunzhong07/Random-Erasing)：此代码具有“随机擦除数据扩充”一文的源代码。
168. [FaderNetworks](https://github.com/facebookresearch/FaderNetworks)：推子网络：通过滑动属性操纵图像 \- NIPS 2017
169. [FlowNet 2.0](https://github.com/NVIDIA/flownet2-pytorch)：FlowNet 2.0：使用Deep Networks进行光流估计的演变
170. [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)：使用条件GAN合成和操作2048x1024图像tcwang0509.github.io/pix2pixHD
171. [pytorch-smoothgrad](https://github.com/pkdn/pytorch-smoothgrad)：PyTorch中的SmoothGrad实现
172. [RetinaNet](https://github.com/c0nn3r/RetinaNet)：PyTorch中RetinaNet的一个实现。
173. [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)：这个项目是一个更快的R-CNN实现，旨在加速更快的R-CNN对象检测模型的训练。
174. [mixup_pytorch](https://github.com/leehomyc/mixup_pytorch)：PyTorch实现的文章混合：超越PyTorch中的经验风险最小化。
175. [inplace_abn](https://github.com/mapillary/inplace_abn)：用于内存优化的DNN培训的就地激活BatchNorm
176. [pytorch-pose-hg-3d](https://github.com/xingyizhou/pytorch-pose-hg-3d)：用于3D人体姿势估计的PyTorch实现
177. [nmn-pytorch](https://github.com/HarshTrivedi/nmn-pytorch)：[Pytorch](https://github.com/HarshTrivedi/nmn-pytorch)中VQA的神经模块网络。
178. [bytenet](https://github.com/kefirski/bytenet)：来自“线性时间神经机器翻译”论文的[字节网的](https://github.com/kefirski/bytenet) Pytorch实现
179. [自下而上注意力vqa](https://github.com/hengyuan-hu/bottom-up-attention-vqa)：vqa，自下而上注意力，pytorch
180. [yolo2-pytorch](https://github.com/ruiminshen/yolo2-pytorch)：YOLOv2是最受欢迎的单级物体探测器之一。该项目采用PyTorch作为提高生产力的开发框架，并利用ONNX将模型转换为Caffe 2到有利于工程的部署。
181. [reseg-pytorch](https://github.com/Wizaron/reseg-pytorch)：ReSeg的PyTorch实现（arxiv.org/pdf/1511.07053.pdf）
182. [二元随机神经元](https://github.com/Wizaron/binary-stochastic-neurons)：PyTorch中的二元随机神经元。
183. [pytorch-pose-estimation](https://github.com/DavexPro/pytorch-pose-estimation)：实时多人姿态估计项目的PyTorch实现。
184. [interaction\_network\_pytorch](https://github.com/higgsfield/interaction_network_pytorch)：用于学习对象，关系和物理的交互网络的Pytorch实现。
185. [NoisyNaturalGradient](https://github.com/wlwkgus/NoisyNaturalGradient)：Pytorch实施论文“嘈杂的自然梯度作为变分推理”。
186. [ewc.pytorch](https://github.com/moskomule/ewc.pytorch)：James Kirkpatrick等人提出的弹性重量合并（EWC）的实现。克服2016年神经网络中的灾难性遗忘（10.1073 / pnas.1611835114）。
187. [pytorch-zssr](https://github.com/jacobgil/pytorch-zssr)：PyTorch使用深度内部学习实现1712.06087“零射击”超分辨率
188. [deep\_image\_prior](https://github.com/atiyo/deep_image_prior)：来自[PyTorch的](https://github.com/atiyo/deep_image_prior) Deep Image Prior（Ulyanov等，2017）的图像重建方法的实现。
189. [pytorch-transformer](https://github.com/leviswind/pytorch-transformer)：pytorch实现注意就是你所需要的。
190. [DeepRL-Grounding](https://github.com/devendrachaplot/DeepRL-Grounding)：这是针对任务导向语言接地的AAAI-18纸张门控注意架构的PyTorch实现
191. [deep-forecast-pytorch](https://github.com/Wizaron/deep-forecast-pytorch)：使用PyTorch中的LSTM进行风速预测（arxiv.org/pdf/1707.08110.pdf）
192. [cat-net](https://github.com/utiasSTARS/cat-net)：Canonical Appearance Transformations
193. [minimal_glo](https://github.com/tneumann/minimal_glo)：从“优化生成网络的潜在空间”一文中生成潜在优化的最小PyTorch实现
194. [LearningToCompare-Pytorch](https://github.com/dragen1860/LearningToCompare-Pytorch)：论文的Pytorch实现：学习比较：少数射击学习的关系网络。
195. [poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings)：PyTorch实施NIPS-17论文“PoincaréEmbeddingsfor Learning Hierarchical Representations”。
196. [pytorch-trpo（Hessian-vector product version）](https://github.com/ikostrikov/pytorch-trpo)：这是PyTorch实现的“信任区域策略优化（TRPO）”，具有精确的Hessian向量积而不是有限差分近似。
197. [ggnn.pytorch](https://github.com/JamesChuanggg/ggnn.pytorch)：门控图序列神经网络（GGNN）的PyTorch实现。
198. [visual-interaction-networks-pytorch](https://github.com/Mrgemy95/visual-interaction-networks-pytorch)：这是使用pytorch进行深度视觉交互网络的一种实现
199. [adversarial-patch](https://github.com/jhayes14/adversarial-patch)：PyTorch实现的对抗补丁。
200. [Prototypical-Networks-for-Few-shot-Learning-PyTorch](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)：在[Pytorch中](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)实现少数镜头学习的原型网络（arxiv.org/abs/1703.05175）
201. [Visual-Feature-Attribution-Using-Wasserstein-GANs-Pytorch](https://github.com/orobix/Visual-Feature-Attribution-Using-Wasserstein-GANs-Pytorch)：在PyTorch中使用Wasserstein GAN（arxiv.org/abs/1711.08998）实现视觉特征归因。
202. [PhotographicImageSynthesiswithCascadedRefinementNetworks-Pytorch](https://github.com/Blade6570/PhotographicImageSynthesiswithCascadedRefinementNetworks-Pytorch)：使用级联细化网络的摄影图像合成 \- Pytorch实现
203. [ENAS-pytorch](https://github.com/carpedm20/ENAS-pytorch)：PyTorch实现“通过参数共享进行高效的神经架构搜索”。
204. [神经](https://github.com/kentsyx/Neural-IMage-Assessment) IMage [评估](https://github.com/kentsyx/Neural-IMage-Assessment)：神经IMage [评估](https://github.com/kentsyx/Neural-IMage-Assessment)的PyTorch实现。
205. [proxprop](https://github.com/tfrerix/proxprop)：近端反向传播 \- 神经网络训练算法，采用隐式梯度步骤而不是显式梯度步骤。
206. [FastPhotoStyle](https://github.com/NVIDIA/FastPhotoStyle)：照片般逼真的图像样式化的封闭形式解决方案
207. [Deep-Image-Analogy-PyTorch](https://github.com/Ben-Louis/Deep-Image-Analogy-PyTorch)：基于pytorch的Deep-Image-Analogy的python实现。
208. [Person-reID_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)：PyTorch for Person re-ID。
209. [pt-dilate-rnn](https://github.com/zalandoresearch/pt-dilate-rnn)：[pytorch中的](https://github.com/zalandoresearch/pt-dilate-rnn)扩张RNN。
210. [pytorch-i-revnet](https://github.com/jhjacobsen/pytorch-i-revnet)：[i-RevNets的](https://github.com/jhjacobsen/pytorch-i-revnet) Pytorch实现。
211. [OrthNet](https://github.com/Orcuslc/OrthNet)：用于生成正交多项式的TensorFlow和PyTorch图层。
212. [DRRN-pytorch](https://github.com/jt827859032/DRRN-pytorch)：超分辨率深度递归残差网络（DRRN）的实现，CVPR 2017
213. [shampoo.pytorch](https://github.com/moskomule/shampoo.pytorch)：洗发水的实施。
214. [神经](https://github.com/truskovskiyk/nima.pytorch) IMage [评估2](https://github.com/truskovskiyk/nima.pytorch)：神经IMage [评估](https://github.com/truskovskiyk/nima.pytorch)的PyTorch实现。
215. [TCN](https://github.com/locuslab/TCN)：序列建模基准和时间卷积网络locuslab / TCN
216. [DCC](https://github.com/shahsohil/DCC)：此存储库包含用于再现Deep Continuous Clustering纸张结果的源代码和数据。
217. [packnet](https://github.com/arunmallya/packnet)：[PackNet](https://github.com/arunmallya/packnet)代码：通过迭代修剪将多个任务添加到单个网络arxiv.org/abs/1711.05769
218. [PyTorch-progressive\_growing\_of_gans](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans)：PyTorch实现GAN的逐步增长，以提高质量，稳定性和变异性。
219. [nonauto-nmt](https://github.com/salesforce/nonauto-nmt)：PyTorch实现“非自回归神经机器翻译”
220. [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)：生成对抗网络的PyTorch实现。
221. [PyTorchWavelets](https://github.com/tomrunia/PyTorchWavelets)：在Torrence和Compo（1998）中发现的小波分析的PyTorch实现
222. [pytorch-made](https://github.com/karpathy/pytorch-made)：[PyTorch中的](https://github.com/karpathy/pytorch-made) MADE（Masked Autoencoder Density Estimation）实现
223. [VRNN](https://github.com/emited/VariationalRecurrentNeuralNetwork)：Pytorch实现变分RNN（VRNN），来自顺序数据的循环潜变量模型。
224. [流程](https://github.com/emited/flow)：Pytorch实施ICLR 2018论文物理过程的深度学习：整合先前的科学知识。
225. [deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch)：PyTorch实现基于卷积网络的文本到语音合成模型
226. [psmm](https://github.com/elanmart/psmm)：Pointer Sentinel混合模型的实现，如Stephen Merity等人的论文中所述。
227. [tacotron2](https://github.com/NVIDIA/tacotron2)：Tacotron 2 - PyTorch实现，实现速度快于实时推理。
228. [AccSGD](https://github.com/rahulkidambi/AccSGD)：实现Accelerated SGD算法的pytorch代码。
229. [QANet-pytorch](https://github.com/hengruo/QANet-pytorch)：使用PyTorch实现QANet（EM / F1 = 70.5 / 77.2，在一个1080Ti卡上20个时刻后约20个小时。）
230. [ConvE](https://github.com/TimDettmers/ConvE)：卷积2D知识图嵌入
231. [结构化自我注意](https://github.com/kaushalshetty/Structured-Self-Attention)：论文的实施结构化自我谦卑句子嵌入，发布于2017年ICLR：arxiv.org/abs/1703.03130。
232. [graphage-simple](https://github.com/williamleif/graphsage-simple)：[GraphSAGE的简单](https://github.com/williamleif/graphsage-simple)参考实现。
233. [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch)：Detectron的pytorch实现。从头开始训练并直接从预训练的Detectron重量推断都是可用的。
234. [R2Plus1D-PyTorch](https://github.com/irhumshafkat/R2Plus1D-PyTorch)：基于R2Plus1D卷积的ResNet架构的PyTorch实现在文章“仔细研究动态识别的时空卷积”中有所描述
235. [StackNN](https://github.com/viking-sudo-rm/StackNN)：用于神经网络的可区分堆栈的PyTorch实现。
236. [translagent](https://github.com/facebookresearch/translagent)：多Agent通信中的紧急翻译代码。
237. [ban-vqa](https://github.com/jnhwkim/ban-vqa)：用于视觉问答的双线性注意网络。
238. [pytorch-openai-transformer-lm](https://github.com/huggingface/pytorch-openai-transformer-lm)：这是TensorFlow代码的PyTorch实现，该代码由OpenAI的论文“由Generative Pre-Training提高语言理解”提供，由Alec Radford，Karthik Narasimhan，Tim Salimans和Ilya Sutskever提供。
239. [T2F](https://github.com/akanimax/T2F)：使用深度学习生成文本到面。该项目结合了最近的两个架构StackGAN和ProGAN，用于从文本描述中合成面部。
240. [pytorch - fid](https://github.com/mseitzer/pytorch-fid)：[PyTorch](https://github.com/mseitzer/pytorch-fid)的Fréchet初始距离（FID得分）
241. [vae_vpflows](https://github.com/jmtomczak/vae_vpflows)：PyTorch中用于凸组合线性IAF和Householder Flow的代码，JM Tomczak和M. Welling jmtomczak.github.io/deebmed.html
242. [CoordConv-pytorch](https://github.com/mkocabas/CoordConv-pytorch)：CoordConv的Pytorch实现在'卷入神经网络的有趣失败和CoordConv解决方案'论文中引入。（arxiv.org/pdf/1807.03247.pdf）
243. [SDPoint](https://github.com/xternalz/SDPoint)：在CVPR 2018中发布的“卷积网络中用于成本可调推理和改进正则化的随机下采样”的实施。
244. [SRDenseNet-pytorch](https://github.com/wxywhu/SRDenseNet-pytorch)：SRDenseNet-pytorch（ICCV_2017）
245. [GAN_stability](https://github.com/LMescheder/GAN_stability)：论文代码“GAN实际上哪些训练方法能够融合？（ICML 2018）”
246. [Mask-RCNN](https://github.com/wannabeOG/Mask-RCNN)：Mask RCNN架构的PyTorch实现，作为使用PyTorch的介绍
247. [pytorch-coviar](https://github.com/chaoyuaw/pytorch-coviar)：压缩视频动作识别
248. [PNASNet.pytorch](https://github.com/chenxi116/PNASNet.pytorch)：ImageNet上的PNASNet-5的PyTorch实现。
249. [NALU-pytorch](https://github.com/kevinzakka/NALU-pytorch)：来自神经算术逻辑单元的NAC / NALU的基本pytorch实现arxiv.org/pdf/1808.00508.pdf
250. [LOLA_DiCE](https://github.com/alexis-jacq/LOLA_DiCE)：Pytorch使用DiCE实现LOLA（arxiv.org/abs/1709.04326）（arxiv.org/abs/1802.05098）
251. [generative-query-network-pytorch](https://github.com/wohlert/generative-query-network-pytorch)：[PyTorch中的](https://github.com/wohlert/generative-query-network-pytorch)生成查询网络（GQN），如“神经场景表示和渲染”中所述
252. [pytorch_hmax](https://github.com/wmvanvliet/pytorch_hmax)：在PyTorch中实现HMAX视觉模型。
253. [FCN-pytorch-easiest](https://github.com/yunlongdong/FCN-pytorch-easiest)：尝试成为FCN（完全控制网络）中最简单，最简单的pytorch实现
254. [传感器](https://github.com/awni/transducer)：使用PyTorch绑定实现快速序列传感器。
255. [AVO-pytorch](https://github.com/artix41/AVO-pytorch)：[PyTorch](https://github.com/artix41/AVO-pytorch)中对抗变分优化的实现。
256. [HCN-pytorch](https://github.com/huguyuehuhu/HCN-pytorch)：{for-occurrence特征学习从骨架数据学习用于动作识别和分层聚合检测}的pytorch重新实现}。
257. [binary-wide-resnet](https://github.com/szagoruyko/binary-wide-resnet)：McDonnel的具有1位权重的宽剩余网络的PyTorch实现（ICLR 2018）
258. [搭载](https://github.com/arunmallya/piggyback)：[背驮式](https://github.com/arunmallya/piggyback)代码：通过学习掩盖权重使单个网络适应多个任务arxiv.org/abs/1801.06519
259. [vid2vid](https://github.com/NVIDIA/vid2vid)：Pytorch实现我们的高分辨率（例如2048x1024）逼真的视频到视频转换方法。
260. [poisson-convolution-sum](https://github.com/cranmer/poisson-convolution-sum)：实现一个无限的泊松加权卷积和
261. [tbd-nets](https://github.com/davidmascharka/tbd-nets)：PyTorch实现“透明设计：缩小视觉推理中性能和可解释性之间的差距”arxiv.org/abs/1803.05268
262. [attn2d](https://github.com/elbayadm/attn2d)：普遍关注：用于序列到序列预测的2D卷积网络
263. [yolov3](https://github.com/ultralytics/yolov3)：[YOLOv3](https://github.com/ultralytics/yolov3)：PyTorch的培训和推理pjreddie.com/darknet/yolo
264. [deep-dream-in-pytorch](https://github.com/duc0/deep-dream-in-pytorch)：Pytorch实现的DeepDream计算机视觉算法。
265. [pytorch-flows](https://github.com/ikostrikov/pytorch-flows)：用于密度估计的算法的PyTorch实现
266. [quantile-regression-dqn-pytorch](https://github.com/ars-ashuha/quantile-regression-dqn-pytorch)：分位数回归DQN是最小工作实例
267. [relational-rnn-pytorch](https://github.com/L0SG/relational-rnn-pytorch)：[PyTorch](https://github.com/L0SG/relational-rnn-pytorch)中DeepMind的关系递归神经网络的实现。
268. [DEXTR-PyTorch](https://github.com/scaelles/DEXTR-PyTorch)：Deep Extreme Cut [http://www.vision.ee.ethz.ch/~cvlsegmentation/dextr](http://www.vision.ee.ethz.ch/~cvlsegmentation/dextr)
269. [PyTorch\_GBW\_LM](https://github.com/rdspring1/PyTorch_GBW_LM)：Google Billion Word Dataset的PyTorch语言模型。
270. [Pytorch-NCE](https://github.com/Stonesjtu/Pytorch-NCE)：用Pytorch编写的softmax输出的噪声对比估计
271. [生成模型](https://github.com/shayneobrien/generative-models)：注释，可理解和可视化解释的PyTorch实现：VAE，BIRVAE，NSGAN，MMGAN，WGAN，WGANGP，LSGAN，DRAGAN，BEGAN，RaGAN，InfoGAN，fGAN，FisherGAN。
272. [convnet-aig](https://github.com/andreasveit/convnet-aig)：具有自适应推理图的卷积网络的PyTorch实现。
273. [integrated-gradient-pytorch](https://github.com/TianhongDai/integrated-gradient-pytorch)：这是本文的pytorch实现 - Deep Networks的Axiomatic Attribution。
274. [MalConv-Pytorch](https://github.com/Alexander-H-Liu/MalConv-Pytorch)：Maltorv的Pytorch实现。
275. [trellisnet](https://github.com/locuslab/trellisnet)：用于序列建模的Trellis Networks
276. [学习与深度多智能体强化学习交流](https://github.com/minqi/learning-to-communicate-pytorch)：学习与深度多智能体强化学习论文交流的pytorch实现。
277. [pnn.pytorch](https://github.com/michaelklachko/pnn.pytorch)：CVPR'18的PyTorch实现 - 扰动神经网络[http://xujuefei.com/pnn.html](http://xujuefei.com/pnn.html)。
278. [Face\_Attention\_Network](https://github.com/rainofmine/Face_Attention_Network)：面部注意网络的Pytorch实现，如面部注意网络：封闭面部的有效面部检测器中所述。
279. [waveglow](https://github.com/NVIDIA/waveglow)：基于流的语音合成生成网络。
280. [deepfloat](https://github.com/facebookresearch/deepfloat)：此存储库包含SystemVerilog RTL，C ++，HLS（用于包装RTL代码的英特尔FPGA OpenCL）和重现“重新思考深度学习浮点数”中的数值结果所需的Python
281. [EPSR](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw)：[使用增强的感知超分辨率网络分析感知 \- 失真权衡的](https://arxiv.org/pdf/1811.00344.pdf) Pytorch实现。作为ECCV 2018的一部分，这项工作在PIRM2018-SR竞赛（地区1）中获得了第一名。
282. [ClariNet](https://github.com/ksw0306/ClariNet)：ClariNet的Pytorch实现arxiv.org/abs/1807.07281
283. [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)：PyTorch版本的Google AI的BERT模型，带有加载Google预训练模型的脚本
284. [torch_waveglow](https://github.com/npuichigo/waveglow)：WaveGlow的PyTorch实现：基于流的语音合成生成网络。
285. [3DDFA](https://github.com/cleardusk/3DDFA)：pytorch改进了TPAMI 2017论文的重新实施：全姿态范围内的面部对齐：3D整体解决方案。
286. [loss-landscape](https://github.com/tomgoldstein/loss-landscape)：loss-landscape用于可视化神经网络损失情况的代码。
287. [famos](https://github.com/zalandoresearch/famos)：Pytorch实施的文章“复制旧的或重新绘制？（非）参数图像样式的对抗框架”，可在[http://arxiv.org/abs/1811.09236获得](http://arxiv.org/abs/1811.09236)。
288. [back2future.pytorch](https://github.com/anuragranj/back2future.pytorch)：这是Janai，J.，Güney，F.，Ranjan，A.，Black，M。和Geiger，A。，Non -pervised Learning of Multi-Frame Optical Flow with Occlusions的Pytorch实现。ECCV 2018。
289. [FFTNet](https://github.com/mozilla/FFTNet)：FFTNet声码文件的非官方实现。
290. [FaceBoxes.PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch)：FaceBoxes的PyTorch实现。
291. [Transformer-XL](https://github.com/kimiyoung/transformer-xl)：Transformer-XL：超越固定长度的语言模型Contexthttps：//github.com/kimiyoung/transformer-xl
292. [associative\_compression\_networks](https://github.com/jalexvig/associative_compression_networks)：用于表示学习的关联压缩网络。
293. [fluidnet_cxx](https://github.com/jolibrain/fluidnet_cxx)：使用ATen张量库重写FluidNet。
294. [Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)：此存储库包含深度强化学习算法的PyTorch实现。
295. [Shufflenet-v2-Pytorch](https://github.com/ericsun99/Shufflenet-v2-Pytorch)：这是faceplusplus的ShuffleNet-v2的Pytorch实现。
296. [GraphWaveletNeuralNetwork](https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork)：这是图形小波神经网络的Pytorch实现。ICLR 2019。
297. [AttentionWalk](https://github.com/benedekrozemberczki/AttentionWalk)：这是Pytorch实现的Watch Your Step：通过Graph Attention学习节点嵌入。NIPS 2018。
298. [SGCN](https://github.com/benedekrozemberczki/SGCN)：这是签名图卷积网络的Pytorch实现。ICDM 2018。
299. [SINE](https://github.com/benedekrozemberczki/SINE)：这是[SINE](https://github.com/benedekrozemberczki/SINE)的Pytorch实现：可扩展的不完整网络嵌入。ICDM 2018。
300. [GAM](https://github.com/benedekrozemberczki/GAM)：这是使用结构注意的图表分类的Pytorch实现。KDD 2018。
301. [neural-style-pt](https://github.com/ProGamerGov/neural-style-pt)：Justin Johnson的神经风格的PyTorch实现。
302. [TuckER](https://github.com/ibalazevic/TuckER)：TuckER：知识图完成的张量分解。
303. [pytorch-prunes](https://github.com/BayesWatch/pytorch-prunes)：修剪神经网络：是时候把它扼杀在萌芽状态了吗？
304. [SimGNN](https://github.com/benedekrozemberczki/SimGNN)：SimGNN：一种快速图形相似度计算的神经网络方法。
305. [字符CNN](https://github.com/ahmedbesbes/character-based-cnn)：用于文本分类的字符级卷积网络的PyTorch实现。
306. [XLM](https://github.com/facebookresearch/XLM)：PyTorch原始实现的跨语言模型预训练。
307. [DiffAI](https://github.com/eth-sri/diffai)：对抗对抗性示例和用于构建兼容PyTorch模型的库的可证明的防御。
308. [APPNP](https://github.com/benedekrozemberczki/APPNP)：将神经网络与个性化PageRank结合起来进行图表分类。ICLR 2019。
309. [NGCN](https://github.com/benedekrozemberczki/MixHop-and-N-GCN)：高阶图卷积层。NeurIPS 2018。
310. [gpt-2-Pytorch](https://github.com/graykode/gpt-2-Pytorch)：使用OpenAI gpt-2 Pytorch实现的简单文本生成器
311. [Splitter](https://github.com/benedekrozemberczki/Splitter)：Splitter：捕获多个社交上下文的学习节点表示。（WWW 2019）。
312. [CapsGNN](https://github.com/benedekrozemberczki/CapsGNN)：胶囊图神经网络。（ICLR 2019）。
313. [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch)：作者的正式非官方PyTorch BigGAN实现。
314. [ppo\_pytorch\_cpp](https://github.com/mhubii/ppo_pytorch_cpp)：这是[Pytorch](https://github.com/mhubii/ppo_pytorch_cpp)的C ++ API的近端策略优化算法的实现。
315. [RandWireNN](https://github.com/seungwonpark/RandWireNN)：实施：“探索随机连接的神经网络进行图像识别”。
316. [Zero-shot Intent CapsNet](https://github.com/joel-huang/zeroshot-capsnet-pytorch)：GPU加速的PyTorch实现“通过胶囊神经网络进行零射击用户意图检测”。
317. [SEAL-CI](https://github.com/benedekrozemberczki/SEAL-CI)半监督图分类：分层图透视。（WWW 2019）。
318. [MixHop](https://github.com/benedekrozemberczki/MixHop-and-N-GCN)：MixHop：通过稀疏邻域混合的高阶图形卷积体系结构。ICML 2019。
319. [densebody_pytorch](https://github.com/Lotayou/densebody_pytorch)：PyWorch实现了CloudWalk最近的论文DenseBody。
320. [voicefilter](https://github.com/mindslab-ai/voicefilter)：非官方的PyTorch实现Google AI的VoiceFilter系统[http://swpark.me/voicefilter](http://swpark.me/voicefilter)。
321. [NVIDIA /语义分段](https://github.com/NVIDIA/semantic-segmentation)：在CVPR2019中[通过视频传播和标签松弛改进语义分割的](https://arxiv.org/abs/1812.01593) PyTorch实现。
322. [ClusterGCN](https://github.com/benedekrozemberczki/ClusterGCN)：PyTorch实现的“Cluster-GCN：一种用于训练深度和大型图形卷积网络的有效算法”（KDD 2019）。

## [](https://github.com/qbmzc/Awesome-pytorch-list#pytorch-elsewhere)Pytorch在其他地方

1. **[the-incredible-pytorch](https://github.com/ritchieng/the-incredible-pytorch)**：The Incredible PyTorch：精选的教程，论文，项目，社区以及与PyTorch相关的更多内容。
2. [生成模型](https://github.com/wiseodd/generative-models)：[生成模型的](https://github.com/wiseodd/generative-models)集合，例如Tensorflow，Keras和Pytorch中的GAN，VAE。[http://wiseodd.github.io](http://wiseodd.github.io/)
3. [pytorch vs tensorflow](https://www.reddit.com/r/MachineLearning/comments/5w3q74/d_so_pytorch_vs_tensorflow_whats_the_verdict_on/)：reddit上的一个信息性线程。
4. [Pytorch讨论论坛](https://discuss.pytorch.org/)
5. [pytorch notebook：docker-stack](https://hub.docker.com/r/escong/pytorch-notebook/)：类似于[Jupyter Notebook Scientific Python Stack的项目](https://github.com/jupyter/docker-stacks/tree/master/scipy-notebook)
6. [drawlikebobross](https://github.com/kendricktan/drawlikebobross)：使用神经网络（使用PyTorch）的力量像Bob Ross一样绘制！
7. [pytorch-tvmisc](https://github.com/t-vi/pytorch-tvmisc)：完全用于Pytorch的多功能杂项
8. [pytorch-a3c-mujoco](https://github.com/andrewliao11/pytorch-a3c-mujoco)：为Mujoco健身房环境实施A3C。
9. [PyTorch在5分钟内完成](https://www.youtube.com/watch?v=nbJ-2G2GXL0&list=WL&index=9)。
10. [pytorch_chatbot](https://github.com/jinfagang/pytorch_chatbot)：使用PyTorch实现的奇妙ChatBot。
11. [malmo-challenge](https://github.com/Kaixhin/malmo-challenge)：Malmo Collaborative AI Challenge - Team Pig Catcher
12. [sketchnet](https://github.com/jtoy/sketchnet)：一种模型，它采用图像并生成处理源代码以重新生成该图像
13. [Deep-Learning-Boot-Camp](https://github.com/QuantScientist/Deep-Learning-Boot-Camp)：一个非营利性社区运行，为期5天的深度学习[训练营](https://github.com/QuantScientist/Deep-Learning-Boot-Camp)[http://deep-ml.com](http://deep-ml.com/)。
14. [Amazon\_Forest\_Computer_Vision](https://github.com/mratsim/Amazon_Forest_Computer_Vision)：使用PyTorch / Keras和许多PyTorch技巧的卫星图像标记代码。讨价还价的比赛。
15. [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)：Gomoku的AlphaZero算法的实现（也称为Gobang或连续五个）
16. [pytorch-cv](https://github.com/youansheng/pytorch-cv)：物体检测，分割和姿态估计的回购。
17. [deep-person-](https://github.com/KaiyangZhou/deep-person-reid) reid：Pytorch实施深层人员重新识别方法。
18. [pytorch-template](https://github.com/victoresque/pytorch-template)：PyTorch模板项目
19. [使用Pytorch TextBook进行深度学习使用PyTorch](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-pytorch)在文本和视觉中构建神经网络模型的实用指南。[在亚马逊](https://www.amazon.in/Deep-Learning-PyTorch-practical-approach/dp/1788624335/ref=tmm_pap_swatch_0?_encoding=UTF8&qid=1523853954&sr=8-1) [github代码回购](https://github.com/svishnu88/DLwithPyTorch)[购买](https://www.amazon.in/Deep-Learning-PyTorch-practical-approach/dp/1788624335/ref=tmm_pap_swatch_0?_encoding=UTF8&qid=1523853954&sr=8-1)[](https://github.com/svishnu88/DLwithPyTorch)
20. [compare-tensorflow-pytorch](https://github.com/jalola/compare-tensorflow-pytorch)：比较用Tensorflow写的层和用Pytorch写的层之间的输出。
21. [hasktorch](https://github.com/hasktorch/hasktorch)：Haskell中的张量和神经网络
22. [Pytorch深度学习使用PyTorch进行](https://www.manning.com/books/deep-learning-with-pytorch)深度学习教你如何使用Python和PyTorch实现深度学习算法。
23. [nimtorch](https://github.com/fragcolor-xyz/nimtorch)：PyTorch - Python + Nim
24. [derplearning](https://github.com/John-Ellis/derplearning)：自驾车RC码。
25. [pytorch-saltnet](https://github.com/tugstugi/pytorch-saltnet)：Kaggle | 针对TGS盐识别挑战的第9位单一模型解决方案。
26. [pytorch-scripts](https://github.com/peterjc123/pytorch-scripts)：[PyTorch的](https://github.com/peterjc123/pytorch-scripts)一些Windows特定脚本。
27. [pytorch_misc](https://github.com/ptrblck/pytorch_misc)：为PyTorch讨论板创建的代码片段。
28. [awesome-pytorch-scholarship](https://github.com/arnas/awesome-pytorch-scholarship)：一系列令人敬畏的PyTorch奖学金文章，指南，博客，课程和其他资源。
29. [MentisOculi](https://github.com/mmirman/MentisOculi)：用PyTorch编写的光线跟踪器（raynet？）
30. [DoodleMaster](https://github.com/karanchahal/DoodleMaster)：“不要编写你的UI代码，画它！”
31. [ocaml-torch](https://github.com/LaurentMazare/ocaml-torch)：PyTorch的OCaml绑定。
32. [extension-script](https://github.com/pytorch/extension-script)：TorchScript的自定义C ++ / CUDA运算符的示例存储库。
33. [pytorch-inference](https://github.com/zccyman/pytorch-inference)：在Windows10平台上用C ++进行PyTorch 1.0推理。
34. [pytorch-cpp-inference](https://github.com/Wizaron/pytorch-cpp-inference)：在C ++中将PyTorch 1.0模型用作Web服务器。
35. [tch-rs](https://github.com/LaurentMazare/tch-rs)：PyTorch的Rust绑定。

##### [](https://github.com/qbmzc/Awesome-pytorch-list#feedback-if-you-have-any-ideas-or-you-want-any-other-content-to-be-added-to-this-list-feel-free-to-contribute)反馈：如果您有任何想法或者您希望将任何其他内容添加到此列表中，请随时提供。
