%from 冯鹏飞
%email：571428374@qq.com & fpf0103@163.com
%time:20241216
% Readme: This code is an experiment in ablation modeling 
% after eliminating the depth-separable convolution module 
% from the multimodal model
% 说明：该代码是取消掉多模态模型中深度可分离卷积模块后的消融模型实验
clc
clear

% Set the random seed for reproducibility
rng(42);
% Set the GPU device and disable cuDNN non-deterministic algorithms
% gpuDevice(1); % Select the first GPU
parallel.gpu.rng(42, 'Philox'); % Set GPU random seed
%%
%加载数字样本数据作为图像数据存储。imageDatastore 根据文件夹名称自动标注图像，
slope="../slope"; %训练集名称
aspect="../aspect"; %集名称
T227="../227";%集名称

imsslope = imageDatastore(slope,'IncludeSubfolders',true,'FileExtensions','.tiff','LabelSource','foldernames');
[imdsTrain_slope,imdsValidation_slope] = splitEachLabel(imsslope,0.9);
imsaspect = imageDatastore(aspect,'IncludeSubfolders',true,'FileExtensions','.tiff','LabelSource','foldernames');
[imdsTrain_aspect,imdsValidation_aspect] = splitEachLabel(imsaspect,0.9);
imsT227 = imageDatastore(T227,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
[imdsTrain_T227,imdsValidation_T227] = splitEachLabel(imsT227,0.9);
arrdsTrain = arrayDatastore(imdsTrain_T227.Labels);
arrdsValidation = arrayDatastore(imdsValidation_T227.Labels);

imdsTrain = combine(imdsTrain_T227,imdsTrain_aspect,imdsTrain_slope,arrdsTrain);

imdsValidation = combine(imdsValidation_T227,imdsValidation_aspect,imdsValidation_slope,arrdsValidation);
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

    concatenationLayer(3,3, 'Name' , 'concat' )

    %aspc模块
    swishLayer("Name","aspc_relu")
    convolution2dLayer([1 1],128,'Stride' ,1,"Name","aspc_conv_1","Padding","same")
    swishLayer("Name","aspc_relu_1")
    depthConcatenationLayer(4, 'Name' , 'aspc_concat' )

    convolution2dLayer([3 3],64,'Stride',1,"Name","conv_E1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_E1")
    swishLayer("Name","relu_E1")

    %分组卷积模块
    % groupedConvolution2dLayer(3,4,'channel-wise','Name','cwconv11_E2',"Padding",[1 1 1 1])
    % swishLayer("Name","relu_E2")
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

%模块二
tempLayers = [
    imageInputLayer([227 227 1],"Name","imageinput_2")
    convolution2dLayer([7 7],64,'Stride' ,2,"Name","conv_2","Padding",[3 3 3 3])
    batchNormalizationLayer("Name","batchnorm_2")
    swishLayer("Name","relu_2")
    maxPooling2dLayer(3,'Stride',2,"Name","pool_2","Padding",[1 1 1 1])
    

    convolution2dLayer([3 3],64,"Name","fire2-1-squeeze1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-1-batchnorm-squeeze1*1")
    swishLayer("Name","fire2-1-relu-squeeze1*1")
    convolution2dLayer([3 3],64,"Name","fire2-1-squeeze1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-1-batchnorm-squeeze1*1-1")
    additionLayer(2,"Name","fire2-1-concat-1")
    swishLayer("Name","fire2-1-relu-squeeze1*1-1")

    convolution2dLayer([3 3],64,"Name","fire2-1-expand1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-1-batchnorm-expand1*1")
    swishLayer("Name","fire2-1-relu-expand1*1")
    convolution2dLayer([3 3],64,"Name","fire2-1-expand1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-1-batchnorm-expand1*1-1")
    additionLayer(2,"Name","fire2-1-concat")
    swishLayer("Name","fire2-1-relu-expand1*1-1")

    % 第二段
    convolution2dLayer([3 3],128,'Stride' ,2,"Name","fire2-2-squeeze1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-2-batchnorm-squeeze1*1")
    swishLayer("Name","fire2-2-relu-squeeze1*1")
    convolution2dLayer([3 3],128,'Stride' ,1,"Name","fire2-2-squeeze1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-2-batchnorm-squeeze1*1-1")
    additionLayer(2,"Name","fire2-2-concat-1")
    swishLayer("Name","fire2-2-relu-squeeze1*1-1")

    convolution2dLayer([3 3],128,'Stride' ,1,"Name","fire2-2-expand1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-2-batchnorm-expand1*1")
    swishLayer("Name","fire2-2-relu-expand1*1")
    convolution2dLayer([3 3],128,"Name","fire2-2-expand1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-2-batchnorm-expand1*1-1")
    additionLayer(2,"Name","fire2-2-concat")
    swishLayer("Name","fire2-2-relu-expand1*1-1")

    % 第三段
    convolution2dLayer([3 3],256,'Stride' ,2,"Name","fire2-3-squeeze1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-3-batchnorm-squeeze1*1")
    swishLayer("Name","fire2-3-relu-squeeze1*1")
    convolution2dLayer([3 3],256,'Stride' ,1,"Name","fire2-3-squeeze1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-3-batchnorm-squeeze1*1-1")
    additionLayer(2,"Name","fire2-3-concat-1")
    swishLayer("Name","fire2-3-relu-squeeze1*1-1")

    convolution2dLayer([3 3],256,'Stride' ,1,"Name","fire2-3-expand1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-3-batchnorm-expand1*1")
    swishLayer("Name","fire2-3-relu-expand1*1")
    convolution2dLayer([3 3],256,"Name","fire2-3-expand1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire2-3-batchnorm-expand1*1-1")
    additionLayer(2,"Name","fire2-3-concat")
    ];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"pool_2","fire2-1-concat-1/in2");
lgraph = connectLayers(lgraph,"fire2-1-relu-squeeze1*1-1","fire2-1-concat/in2");

fire2_2Layers = [
    convolution2dLayer([1 1],128,'Stride' ,2,"Name","fire2-2-squeeze3*3","Padding",[0 0 0 0])
    batchNormalizationLayer("Name","fire2-2-batchnorm-squeeze3*3")
    ];
lgraph = addLayers(lgraph,fire2_2Layers);
lgraph = connectLayers(lgraph,"fire2-1-relu-expand1*1-1","fire2-2-squeeze3*3");
lgraph = connectLayers(lgraph,"fire2-2-batchnorm-squeeze3*3","fire2-2-concat-1/in2");
lgraph = connectLayers(lgraph,"fire2-2-relu-squeeze1*1-1","fire2-2-concat/in2");

fire2_3Layers = [
    convolution2dLayer([1 1],256,'Stride' ,2,"Name","fire2-3-squeeze3*3","Padding",[0 0 0 0])
    batchNormalizationLayer("Name","fire2-3-batchnorm-squeeze3*3")
    ];
lgraph = addLayers(lgraph,fire2_3Layers);
lgraph = connectLayers(lgraph,"fire2-2-relu-expand1*1-1","fire2-3-squeeze3*3");
lgraph = connectLayers(lgraph,"fire2-3-batchnorm-squeeze3*3","fire2-3-concat-1/in2");
lgraph = connectLayers(lgraph,"fire2-3-relu-squeeze1*1-1","fire2-3-concat/in2");

lgraph = connectLayers(lgraph,"fire2-3-concat","concat/in2");

%模块三
tempLayers3 = [
    imageInputLayer([227 227 1],"Name","imageinput_3")
    convolution2dLayer([7 7],64,'Stride' ,2,"Name","conv_3","Padding",[3 3 3 3])
    batchNormalizationLayer("Name","batchnorm_3")
    swishLayer("Name","relu_3")
    maxPooling2dLayer(3,'Stride',2,"Name","pool_3","Padding",[1 1 1 1])

    convolution2dLayer([3 3],64,"Name","fire3-1-squeeze1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-1-batchnorm-squeeze1*1")
    swishLayer("Name","fire3-1-relu-squeeze1*1")
    convolution2dLayer([3 3],64,"Name","fire3-1-squeeze1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-1-batchnorm-squeeze1*1-1")
    additionLayer(2,"Name","fire3-1-concat-1")
    swishLayer("Name","fire3-1-relu-squeeze1*1-1")

    convolution2dLayer([3 3],64,"Name","fire3-1-expand1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-1-batchnorm-expand1*1")
    swishLayer("Name","fire3-1-relu-expand1*1")
    convolution2dLayer([3 3],64,"Name","fire3-1-expand1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-1-batchnorm-expand1*1-1")
    additionLayer(2,"Name","fire3-1-concat")
    swishLayer("Name","fire3-1-relu-expand1*1-1")

    % 第二段
    convolution2dLayer([3 3],128,'Stride' ,2,"Name","fire3-2-squeeze1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-2-batchnorm-squeeze1*1")
    swishLayer("Name","fire3-2-relu-squeeze1*1")
    convolution2dLayer([3 3],128,'Stride' ,1,"Name","fire3-2-squeeze1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-2-batchnorm-squeeze1*1-1")
    additionLayer(2,"Name","fire3-2-concat-1")
    swishLayer("Name","fire3-2-relu-squeeze1*1-1")

    convolution2dLayer([3 3],128,'Stride' ,1,"Name","fire3-2-expand1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-2-batchnorm-expand1*1")
    swishLayer("Name","fire3-2-relu-expand1*1")
    convolution2dLayer([3 3],128,"Name","fire3-2-expand1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-2-batchnorm-expand1*1-1")
    additionLayer(2,"Name","fire3-2-concat")
    swishLayer("Name","fire3-2-relu-expand1*1-1")

    % 第三段
    convolution2dLayer([3 3],256,'Stride' ,2,"Name","fire3-3-squeeze1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-3-batchnorm-squeeze1*1")
    swishLayer("Name","fire3-3-relu-squeeze1*1")
    convolution2dLayer([3 3],256,'Stride' ,1,"Name","fire3-3-squeeze1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-3-batchnorm-squeeze1*1-1")
    additionLayer(2,"Name","fire3-3-concat-1")
    swishLayer("Name","fire3-3-relu-squeeze1*1-1")

    convolution2dLayer([3 3],256,'Stride' ,1,"Name","fire3-3-expand1*1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-3-batchnorm-expand1*1")
    swishLayer("Name","fire3-3-relu-expand1*1")
    convolution2dLayer([3 3],256,"Name","fire3-3-expand1*1-1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","fire3-3-batchnorm-expand1*1-1")
    additionLayer(2,"Name","fire3-3-concat")
    ];
lgraph = addLayers(lgraph,tempLayers3);

lgraph = connectLayers(lgraph,"pool_3","fire3-1-concat-1/in2");
lgraph = connectLayers(lgraph,"fire3-1-relu-squeeze1*1-1","fire3-1-concat/in2");

fire3_2Layers = [
    convolution2dLayer([1 1],128,'Stride' ,2,"Name","fire3-2-squeeze3*3","Padding",[0 0 0 0])
    batchNormalizationLayer("Name","fire3-2-batchnorm-squeeze3*3")
    ];
lgraph = addLayers(lgraph,fire3_2Layers);
lgraph = connectLayers(lgraph,"fire3-1-relu-expand1*1-1","fire3-2-squeeze3*3");
lgraph = connectLayers(lgraph,"fire3-2-batchnorm-squeeze3*3","fire3-2-concat-1/in2");
lgraph = connectLayers(lgraph,"fire3-2-relu-squeeze1*1-1","fire3-2-concat/in2");

fire3_3Layers = [
    convolution2dLayer([1 1],256,'Stride' ,2,"Name","fire3-3-squeeze3*3","Padding",[0 0 0 0])
    batchNormalizationLayer("Name","fire3-3-batchnorm-squeeze3*3")
    ];
lgraph = addLayers(lgraph,fire3_3Layers);
lgraph = connectLayers(lgraph,"fire3-2-relu-expand1*1-1","fire3-3-squeeze3*3");
lgraph = connectLayers(lgraph,"fire3-3-batchnorm-squeeze3*3","fire3-3-concat-1/in2");
lgraph = connectLayers(lgraph,"fire3-3-relu-squeeze1*1-1","fire3-3-concat/in2");

lgraph = connectLayers(lgraph,"fire3-3-concat","concat/in3");

% 清理辅助变量
clear tempLayers;
clear tempLayers3;
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
save('Final_NonGroupCov_AC_Swish_trained_model.mat', 'net');
%%
%验证测试数据
% 确保网络处于评估模式
net = resetState(net);
imagePred = classify(net, imdsValidation, 'MiniBatchSize', 64);
imageResult = imdsValidation_T227.Labels;
accuracy = sum(imagePred == imageResult)/numel(imageResult)
disp(['Validation accuracy: ', num2str(accuracy * 100), '%']);
% accuracy = 0.8953 0.8845 0.8917 0.9097 0.8917 0.8773 0.8953 0.8917 0.8809 0.8917
% time = 4：56
% 参数 = 10.4M 167