% from 冯鹏飞
% email：571428374@qq.com
% time:20241201
% Readme : This code is for the visualization of 
% the output activation of each operator layer 
% in the full phase of a multimodal deep learning model
% 说明：该代码是用于多模态深度学习模型全阶段各算子层输出activation的可视化
clc
clear all
%%
% 加载预训练的多模态模型
net = load('Final_AC_Swish_trained_model.mat').net;
lgraph = layerGraph(net);
% analyzeNetwork(lgraph)
%%
% 读取并预处理输入图像
lab = 'fyb860';  % 你可以根据需要改变这个值，读取不同的图片
% img = imread(['..\227\landslide_improve_227\' lab '.png']); 
img = imread(['..\227\non-landslide_improve_227\' lab '.png']); 
% 读取并预处理坡向信息nvitop
% aspect = imread(['..\aspect\landslide_improve_dem_227_aspect\' lab '.tiff']); 
aspect = imread(['..\aspect\non-landslide_improve_dem_227_aspect\' lab '.tiff']); 
if size(aspect, 3) ~= 1
    aspect = reshape(aspect, [227, 227, 1]);
end
% 读取并预处理坡度信息

% slope = imread(['..\slope\landslide_improve_dem_227_slope\' lab '.tiff']);
slope = imread(['..\slope\non-landslide_improve_dem_227_slope\' lab '.tiff']);
if size(slope, 3) ~= 1
    slope = reshape(slope, [227, 227, 1]);
end
[classfn,score] = classify(net,img, aspect, slope);
disp(classfn)
% 指定目标层和分类索引（假设目标层为'conv5'，目标类别索引为1）
% targetLayer = 'gap1';
% targetLayer = 'conv_E3';
% targetLayer = 'cwconv11_E2';
% targetLayer = 'conv_E1';
% targetLayer = 'aspc_concat';
% targetLayer = 'aspc_relu_4';
% targetLayer = 'aspc_relu_3';
% targetLayer = 'aspc_relu_2';
% targetLayer = 'aspc_relu_1';
% targetLayer = 'aspc_conv_4';
% targetLayer = 'aspc_conv_3';
% targetLayer = 'aspc_conv_2';
% targetLayer = 'aspc_conv_1';

% targetLayer = 'concat';
% targetLayer = 'fire3-3-concat';
% targetLayer = 'fire2-3-concat';
% targetLayer = 'fire1-3-concat';

% targetLayer = 'fire3-2-concat';
% targetLayer = 'fire2-2-concat';
% targetLayer = 'fire1-2-concat';

% targetLayer = 'fire3-1-concat';
% targetLayer = 'fire2-1-concat';
% targetLayer = 'fire1-1-concat';

% targetLayer = 'pool_3';
% targetLayer = 'pool_2';
% targetLayer = 'pool_1';
% targetLayer = 'relu_3';
% targetLayer = 'relu_2';
% targetLayer = 'relu_1';


% 生成MM-Grad-CAM
activations = mactivation(net, img, aspect, slope, targetLayer);
activations = extractdata(activations);
% 将Grad-CAM叠加到原始图像上


if size(activations,3) > 10
    num = 2;
else
    num = size(activations,3);
end

% 创建文件夹“结果”
output_folder = lab;
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

for i = 1:num
    activations0 = imresize(activations(:,:,i), [size(img,1) size(img,2)], 'bilinear');
    heatmap = ind2rgb(im2uint8(activations0), jet(256));
    overlayedImg = im2uint8(im2double(img) * 0.5 + heatmap * 0.5);

    figure;
    imshow(activations0,'border','tight','initialmagnification','fit');
    axis normal;
    colormap jet
    hold off;

    % 设置图像大小为5cm*5cm，分辨率为300 DPI
    set(gcf, 'PaperUnits', 'centimeters');
    set(gcf, 'PaperSize', [5, 5]);
    set(gcf, 'PaperPosition', [0, 0, 5, 5]);

    targetLayerT = strrep(targetLayer, '*', '-');
    fileName = sprintf('%s_%d.tif', targetLayerT, i);
    % 构建完整的文件路径
    output_file = fullfile(output_folder, fileName);

    % 保存图像
    print(gcf, output_file, '-dtiff', '-r300');

    disp(['图像已保存到文件夹 "', output_folder, '" 中，文件名为 "', sprintf('%s_%d.tif', targetLayer, i), '"']);
end


% targetLayer = 'gap1';
% targetLayer = 'conv_E3';
% targetLayer = 'cwconv11_E2';
% targetLayer = 'aspc_concat';
% targetLayer = 'aspc_relu_4';
% targetLayer = 'aspc_relu_3';
% targetLayer = 'aspc_relu_2';
% targetLayer = 'aspc_relu_1';
% targetLayer = 'aspc_conv_4';
% targetLayer = 'aspc_conv_3';
% targetLayer = 'aspc_conv_2';
% targetLayer = 'aspc_conv_1';

% targetLayer = 'concat';
% targetLayer = 'fire3-3-concat';
% targetLayer = 'fire2-3-concat';
% targetLayer = 'fire1-3-concat';
% targetLayer = 'fire3-3-batchnorm-expand1*1-1';
% targetLayer = 'fire2-3-batchnorm-expand1*1-1';
% targetLayer = 'fire1-3-batchnorm-expand1*1-1';
% targetLayer = 'fire3-3-relu-expand1*1';
% targetLayer = 'fire2-3-relu-expand1*1';
% targetLayer = 'fire1-3-relu-expand1*1';
% targetLayer = 'fire3-3-batchnorm-expand1*1';
% targetLayer = 'fire2-3-batchnorm-expand1*1';
% targetLayer = 'fire1-3-batchnorm-expand1*1';
% targetLayer = 'fire3-3-batchnorm-squeeze3*3';
% targetLayer = 'fire2-3-batchnorm-squeeze3*3';
% targetLayer = 'fire1-3-batchnorm-squeeze3*3';
% targetLayer = 'fire3-3-batchnorm-squeeze1*1-1';
% targetLayer = 'fire2-3-batchnorm-squeeze1*1-1';
% targetLayer = 'fire1-3-batchnorm-squeeze1*1-1';
% targetLayer = 'fire3-3-relu-squeeze1*1';
% targetLayer = 'fire2-3-relu-squeeze1*1';
% targetLayer = 'fire1-3-relu-squeeze1*1';
% targetLayer = 'fire3-3-batchnorm-squeeze1*1';
% targetLayer = 'fire2-3-batchnorm-squeeze1*1';
% targetLayer = 'fire1-3-batchnorm-squeeze1*1';


% targetLayer = 'fire3-2-concat';
% targetLayer = 'fire2-2-concat';
% targetLayer = 'fire1-2-concat';
% targetLayer = 'fire3-2-batchnorm-expand1*1-1';
% targetLayer = 'fire2-2-batchnorm-expand1*1-1';
% targetLayer = 'fire1-2-batchnorm-expand1*1-1';
% targetLayer = 'fire3-2-relu-expand1*1';
% targetLayer = 'fire2-2-relu-expand1*1';
% targetLayer = 'fire1-2-relu-expand1*1';
% targetLayer = 'fire3-2-batchnorm-expand1*1';
% targetLayer = 'fire2-2-batchnorm-expand1*1';
% targetLayer = 'fire1-2-batchnorm-expand1*1';
% targetLayer = 'fire3-2-batchnorm-squeeze3*3';
% targetLayer = 'fire2-2-batchnorm-squeeze3*3';
% targetLayer = 'fire1-2-batchnorm-squeeze3*3';
% targetLayer = 'fire3-2-batchnorm-squeeze1*1-1';
% targetLayer = 'fire2-2-batchnorm-squeeze1*1-1';
% targetLayer = 'fire1-2-batchnorm-squeeze1*1-1';
% targetLayer = 'fire3-2-relu-squeeze1*1';
% targetLayer = 'fire2-2-relu-squeeze1*1';
% targetLayer = 'fire1-2-relu-squeeze1*1';
% targetLayer = 'fire3-2-batchnorm-squeeze1*1';
% targetLayer = 'fire2-2-batchnorm-squeeze1*1';
% targetLayer = 'fire1-2-batchnorm-squeeze1*1';


% targetLayer = 'fire3-1-concat';
% targetLayer = 'fire2-1-concat';
% targetLayer = 'fire1-1-concat';
% targetLayer = 'fire3-1-batchnorm-expand1*1-1';
% targetLayer = 'fire2-1-batchnorm-expand1*1-1';
% targetLayer = 'fire1-1-batchnorm-expand1*1-1';
% targetLayer = 'fire3-1-relu-expand1*1';
% targetLayer = 'fire2-1-relu-expand1*1';
% targetLayer = 'fire1-1-relu-expand1*1';
% targetLayer = 'fire3-1-batchnorm-expand1*1';
% targetLayer = 'fire2-1-batchnorm-expand1*1';
% targetLayer = 'fire1-1-batchnorm-expand1*1';
% targetLayer = 'fire3-1-batchnorm-squeeze1*1-1';
% targetLayer = 'fire2-1-batchnorm-squeeze1*1-1';
% targetLayer = 'fire1-1-batchnorm-squeeze1*1-1';
% targetLayer = 'fire3-1-relu-squeeze1*1';
% targetLayer = 'fire2-1-relu-squeeze1*1';
% targetLayer = 'fire1-1-relu-squeeze1*1';
% targetLayer = 'fire3-1-batchnorm-squeeze1*1';
% targetLayer = 'fire2-1-batchnorm-squeeze1*1';
% targetLayer = 'fire1-1-batchnorm-squeeze1*1';

% targetLayer = 'pool_3';
% targetLayer = 'pool_2';
% targetLayer = 'pool_1';
% targetLayer = 'relu_3';
% targetLayer = 'relu_2';
% targetLayer = 'relu_1';
% targetLayer = 'batchnorm_3';
% targetLayer = 'batchnorm_2';
% targetLayer = 'batchnorm_1';