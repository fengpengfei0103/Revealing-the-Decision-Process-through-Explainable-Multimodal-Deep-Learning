% from 冯鹏飞
% email：571428374@qq.com & fpf0103@163.com
% time:20241212
% Readme: This code is used for model validation
% 说明：该代码用于模型的验证
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

classes = ["landslide_improve_227" "non-landslide_improve_227"];
%%
%模型加载
net = load('Final_AC_Swish_trained_model9422.mat').net;
lgraph = layerGraph(net);
analyzeNetwork(lgraph)
%%
%验证测试数据
% 确保网络处于评估模式
net = resetState(net);
imagePred = classify(net, imdsValidation, 'MiniBatchSize', 64);
imageResult = imdsValidation_T227.Labels;
accuracy = sum(imagePred == imageResult)/numel(imageResult)
disp(['Validation accuracy: ', num2str(accuracy * 100), '%']);