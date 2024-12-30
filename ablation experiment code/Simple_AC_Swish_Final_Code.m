% from 冯鹏飞
% email：571428374@qq.com & fpf0103@163.com
% time: 20240627
% Readme: The code is an ablation model with 
% only a single mode of optical remote sensing as an input
% 说明：该代码是仅有光学遥感单一模态的作为输入的消融模型
clc
clear

% Set the random seed for reproducibility
rng(42);
% Set the GPU device and disable cuDNN non-deterministic algorithms
% gpuDevice(1); % Select the first GPU
parallel.gpu.rng(42, 'Philox'); % Set GPU random seed
%%
%加载数字样本数据作为图像数据存储。imageDatastore 根据文件夹名称自动标注图像，
% slope="../slope"; %训练集名称
% aspect="../aspect"; %集名称
T227="../227";%集名称

% imsslope = imageDatastore(slope,'IncludeSubfolders',true,'FileExtensions','.tiff','LabelSource','foldernames');
% [imdsTrain_slope,imdsValidation_slope] = splitEachLabel(imsslope,0.9);
% imsaspect = imageDatastore(aspect,'IncludeSubfolders',true,'FileExtensions','.tiff','LabelSource','foldernames');
% [imdsTrain_aspect,imdsValidation_aspect] = splitEachLabel(imsaspect,0.9);
imsT227 = imageDatastore(T227,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
[imdsTrain_T227,imdsValidation_T227] = splitEachLabel(imsT227,0.9);
% arrdsTrain = arrayDatastore(imdsTrain_T227.Labels);
% arrdsValidation = arrayDatastore(imdsValidation_T227.Labels);

imdsTrain = imdsTrain_T227;

imdsValidation = imdsValidation_T227;
% imdsValidation = combine(imdsValidation_T227,imdsValidation_aspect,imdsValidation_slope);
classes = ["landslide_improve_227" "non-landslide_improve_227"];
% 使用类别的占比作为权重
classWeights =[2000/2770 770/2770]
%%
%模型构建
%模块一

layers = [
    imageInputLayer([227 227 3],"Name","imageinput_1")
    convolution2dLayer([7 7],64,'Stride' ,2,"Name","conv_1","Padding",[3 3 3 3])
    batchNormalizationLayer("Name","batchnorm_1")
    swishLayer("Name","relu_1")
    maxPooling2dLayer(3,'Stride',2,"Name","pool_1","Padding",[1 1 1 1])
    % tanhLayer leakyReluLayer

    convolution2dLayer([3 3],64,"Name","fire1-1-squeeze1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-1-batchnorm-squeeze1*1")
    swishLayer("Name","fire1-1-relu-squeeze1*1")
    convolution2dLayer([3 3],64,"Name","fire1-1-squeeze1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-1-batchnorm-squeeze1*1-1")
    additionLayer(2,"Name","fire1-1-concat-1")
    swishLayer("Name","fire1-1-relu-squeeze1*1-1")

    convolution2dLayer([3 3],64,"Name","fire1-1-expand1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-1-batchnorm-expand1*1")
    swishLayer("Name","fire1-1-relu-expand1*1")
    convolution2dLayer([3 3],64,"Name","fire1-1-expand1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-1-batchnorm-expand1*1-1")
    additionLayer(2,"Name","fire1-1-concat")
    swishLayer("Name","fire1-1-relu-expand1*1-1")

    % 第二段
    convolution2dLayer([3 3],128,'Stride' ,2,"Name","fire1-2-squeeze1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-2-batchnorm-squeeze1*1")
    swishLayer("Name","fire1-2-relu-squeeze1*1")
    convolution2dLayer([3 3],128,'Stride' ,1,"Name","fire1-2-squeeze1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-2-batchnorm-squeeze1*1-1")
    additionLayer(2,"Name","fire1-2-concat-1")
    swishLayer("Name","fire1-2-relu-squeeze1*1-1")

    convolution2dLayer([3 3],128,'Stride' ,1,"Name","fire1-2-expand1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-2-batchnorm-expand1*1")
    swishLayer("Name","fire1-2-relu-expand1*1")
    convolution2dLayer([3 3],128,"Name","fire1-2-expand1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-2-batchnorm-expand1*1-1")
    additionLayer(2,"Name","fire1-2-concat")
    swishLayer("Name","fire1-2-relu-expand1*1-1")

    % 第三段
    convolution2dLayer([3 3],256,'Stride' ,2,"Name","fire1-3-squeeze1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-3-batchnorm-squeeze1*1")
    swishLayer("Name","fire1-3-relu-squeeze1*1")
    convolution2dLayer([3 3],256,'Stride' ,1,"Name","fire1-3-squeeze1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-3-batchnorm-squeeze1*1-1")
    additionLayer(2,"Name","fire1-3-concat-1")
    swishLayer("Name","fire1-3-relu-squeeze1*1-1")

    convolution2dLayer([3 3],256,'Stride' ,1,"Name","fire1-3-expand1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-3-batchnorm-expand1*1")
    swishLayer("Name","fire1-3-relu-expand1*1")
    convolution2dLayer([3 3],256,"Name","fire1-3-expand1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire1-3-batchnorm-expand1*1-1")
    additionLayer(2,"Name","fire1-3-concat")

    % concatenationLayer(3,3, 'Name' , 'concat' )

    %aspc模块
    swishLayer("Name","aspc_relu")
    convolution2dLayer([1 1],128,'Stride' ,1,"Name","aspc_conv_1","Padding","same")
    swishLayer("Name","aspc_relu_1")
    depthConcatenationLayer(4, 'Name' , 'aspc_concat' )

    convolution2dLayer([3 3],64,'Stride',1,"Name","conv_E1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_E1")
    swishLayer("Name","relu_E1")

    %分组卷积模块
    groupedConvolution2dLayer(3,4,'channel-wise','Name','cwconv11_E2',"Padding",[1 1 1 1])
    swishLayer("Name","relu_E2")
    convolution2dLayer([3 3],2,"Name","conv_E3","Padding",[1 1 1 1])
    globalAveragePooling2dLayer('Name','gap1')

    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput",'Classes',classes,"ClassWeights",classWeights)
];
lgraph = layerGraph(layers);


lgraph = connectLayers(lgraph,"pool_1","fire1-1-concat-1/in2");
lgraph = connectLayers(lgraph,"fire1-1-relu-squeeze1*1-1","fire1-1-concat/in2");
% 
fire1_2Layers = [
    convolution2dLayer([1 1],128,'Stride' ,2,"Name","fire1-2-squeeze3*3","Padding",[0 0 0 0])
    batchNormalizationLayer("Name","fire1-2-batchnorm-squeeze3*3")
    ];
lgraph = addLayers(lgraph,fire1_2Layers);
lgraph = connectLayers(lgraph,"fire1-1-relu-expand1*1-1","fire1-2-squeeze3*3");
lgraph = connectLayers(lgraph,"fire1-2-batchnorm-squeeze3*3","fire1-2-concat-1/in2");
lgraph = connectLayers(lgraph,"fire1-2-relu-squeeze1*1-1","fire1-2-concat/in2");

fire1_3Layers = [
    convolution2dLayer([1 1],256,'Stride' ,2,"Name","fire1-3-squeeze3*3","Padding",[0 0 0 0])
    batchNormalizationLayer("Name","fire1-3-batchnorm-squeeze3*3")
    ];
lgraph = addLayers(lgraph,fire1_3Layers);
lgraph = connectLayers(lgraph,"fire1-2-relu-expand1*1-1","fire1-3-squeeze3*3");
lgraph = connectLayers(lgraph,"fire1-3-batchnorm-squeeze3*3","fire1-3-concat-1/in2");
lgraph = connectLayers(lgraph,"fire1-3-relu-squeeze1*1-1","fire1-3-concat/in2");

aspc2 = [
    convolution2dLayer([3 3],128,'Stride' ,1,'DilationFactor',[2 2],"Name","aspc_conv_2","Padding","same")
    swishLayer("Name","aspc_relu_2")
    ];
lgraph = addLayers(lgraph,aspc2);
lgraph = connectLayers(lgraph,"aspc_relu","aspc_conv_2");
lgraph = connectLayers(lgraph,"aspc_relu_2","aspc_concat/in2");

aspc3 = [
    convolution2dLayer([3 3],128,'Stride' ,1,'DilationFactor',[4 4],"Name","aspc_conv_3","Padding","same")
    swishLayer("Name","aspc_relu_3")
    ];
lgraph = addLayers(lgraph,aspc3);
lgraph = connectLayers(lgraph,"aspc_relu","aspc_conv_3");
lgraph = connectLayers(lgraph,"aspc_relu_3","aspc_concat/in3");

aspc4 = [
    convolution2dLayer([3 3],128,'Stride' ,1,'DilationFactor',[4 4],"Name","aspc_conv_4","Padding","same")
    swishLayer("Name","aspc_relu_4")
    ];
lgraph = addLayers(lgraph,aspc4);
lgraph = connectLayers(lgraph,"aspc_relu","aspc_conv_4");
lgraph = connectLayers(lgraph,"aspc_relu_4","aspc_concat/in4");

analyzeNetwork(lgraph)

%%
%指定训练选项
% Set the GPU device and disable cuDNN non-deterministic algorithms
gpuDevice(1); % Select the first GPU
parallel.gpu.rng(42, 'Philox'); % Set GPU random seed
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'MaxEpochs',20, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',38, ...
    'Verbose',true, ...
    'ExecutionEnvironment','auto',...
    'Plots','training-progress');
%使用训练数据训练网络
[net,info] = trainNetwork(imdsTrain,lgraph,options);
save('Final_Simple_AC_Swish_trained_model.mat', 'net');
%%
%验证测试数据
% 确保网络处于评估模式
net = resetState(net);
imagePred = classify(net, imdsValidation, 'MiniBatchSize', 64);
imageResult = imdsValidation_T227.Labels;
accuracy = sum(imagePred == imageResult)/numel(imageResult)
disp(['Validation accuracy: ', num2str(accuracy * 100), '%']);
% accuracy = 0.8989 0.9134 0.9097 0.9206 0.9025 0.9061 0.8989 0.9206 0.9061 0.8953
% time = 1：03
% 参数 = 4M 69