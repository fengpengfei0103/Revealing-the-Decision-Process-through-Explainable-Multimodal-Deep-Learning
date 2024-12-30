%from 冯鹏飞
%email：571428374@qq.com & fpf0103@163.com
%time:20240627
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
% classes = ["landslide_improve_227" "non-landslide_improve_227"];
% classWeights =[0.5 0.5]
%%
%模型构建
net = vgg19('Weights','none');
lgraph = layerGraph(net);
analyzeNetwork(lgraph)
inputSize = lgraph.Layers(1).InputSize
lgraph = disconnectLayers(lgraph,"input","conv1_1");
lgraph = removeLayers(lgraph,"input");
%%
layers = [
    imageInputLayer([227 227 3],"Name","imageinput_1")
    concatenationLayer(3,3, 'Name' , 'concat' )
];
lgraph = addLayers(lgraph,layers);
%模块二
tempLayers = [
    imageInputLayer([227 227 1],"Name","imageinput_2")
    ];
lgraph = addLayers(lgraph,tempLayers);
lgraph = connectLayers(lgraph,"imageinput_2","concat/in2");

%模块三
tempLayers3 = [
    imageInputLayer([227 227 1],"Name","imageinput_3")
    ];
lgraph = addLayers(lgraph,tempLayers3);
lgraph = connectLayers(lgraph,"imageinput_3","concat/in3");

lgraph = connectLayers(lgraph,"concat","conv1_1");

numClasses = numel(categories(imdsTrain_T227.Labels));
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
    
lgraph = replaceLayer(lgraph,'fc8',newLearnableLayer);
%分类层指定网络的输出类别。用一个没有类别标签的新分类层替换分类层。trainNetwork在训练时自动设置层的输出类。
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);
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
save('trained_model_vgg19.mat', 'net');
%%
%验证测试数据
% 确保网络处于评估模式
net = resetState(net);
imagePred = classify(net, imdsValidation, 'MiniBatchSize', 64);
imageResult = imdsValidation_T227.Labels;
accuracy = sum(imagePred == imageResult)/numel(imageResult)
disp(['Validation accuracy: ', num2str(accuracy * 100), '%']);
% accuracy = 0.7220 0.7220 0.7220 0.7220 0.7220 0.7220 0.7220 0.7220 0.7220 0.7220
% time = 06：22
% 参数 = 139.5M 50