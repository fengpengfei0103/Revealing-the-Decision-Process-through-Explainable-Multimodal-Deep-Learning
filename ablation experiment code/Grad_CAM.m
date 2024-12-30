%from 冯鹏飞
%email：571428374@qq.com
%time:20220817
clc
clear
% Grad- CAM是类激活映射 ( CAM ) 技术的推广。
% 使用梯度加权类激活映射 (Grad- CAM ) 技术来理解深度学习网络做出分类决策的原因
% 使用分类分数相对于网络确定的卷积特征的梯度来了解图像的哪些部分对于分类最重要。

%%
%加载预训练网络
loaded_model = load('Final_Simple_AC_Swish_trained_model.mat');
net = loaded_model.net;
%%
% 图像分类
% 读取图像大小。
inputSize = net.Layers(1).InputSize(1:2);
% 加载图像。
X = imread( "../227/landslide_improve_227/zj050.png" );
% 对图像进行分类并显示它及其分类和分类分数。
[classfn,score] = classify(net,X);
classfn

% Net正确地将图像分类。但为什么？图像的哪些特征导致网络做出这种分类？

%
% Grad- CAM解释原因
% Grad- CAM技术利用分类分数相对于最终卷积特征图的梯度，
% 来识别输入图像中对分类分数影响最大的部分。这个梯度大的地方，正是最终得分最依赖数据的地方。
% gradCAM函数通过求给定类的缩减层输出相对于卷积特征图的导数来计算重要性图。
% 对于分类任务，该gradCAM函数会自动选择合适的层来计算重要性图。
% 'ReductionLayer'您还可以使用和名称-值参数指定图层'FeatureLayer'。

% 计算 Grad- CAM地图
featureLayer = 'aspc_concat';
map = gradCAM(net,X,classfn,'FeatureLayer',featureLayer);
% 使用值 0.5在图像顶部显示 Grad- CAM贴图。'AlphaData'颜色'jet'图的最低值是深蓝色，最高值是深红色。
figure
imshow(X);
imshow(X,'border','tight','initialmagnification','fit');
axis normal;
hold on;
imagesc(map,'AlphaData',0.5);
colormap jet
