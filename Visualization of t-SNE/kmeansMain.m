% from 冯鹏飞
% email：571428374@qq.com
% time:20241227
% Readme : This code is used to visualize the subsequent clustering of 
% the results after recognition
% 说明：该代码是用于识别后结果的后续聚类的可视化
% Image clustering analysis
clc;
clear;
close all
%%

% Step 1: 加载图像数据集
imageFolder = 'landslide_improve_dem_227_aspect'; % 修改为图像数据集所在路径 landslide_improve_227  landslide_improve_dem_227_aspect
imageFiles = dir(fullfile(imageFolder, '*.tiff')); % 修改为图像格式 png tiff
numImages = numel(imageFiles);
features = [];

% Step 2: 特征提取（灰度直方图）
numBins = 16; % 灰度直方图分箱数
for i = 1:numImages
    img = imread(fullfile(imageFolder, imageFiles(i).name));
    if size(img,3) > 1
        imgGray = rgb2gray(img);  % png需要
        histCounts = imhist(imgGray, numBins);
    else
        histCounts = imhist(img, numBins);
    end
    % histCounts = histCounts / sum(histCounts); % 归一化
    features = [features; histCounts']; % 累积特征
end

% Step 3: K均值聚类
numClusters = 5; % 修改为所需聚类数量
[idx, C] = kmeans(features, numClusters);

% Step 4: 统计分析（均值、标准差等）
clusterStats = struct();
for k = 1:numClusters
    clusterData = features(idx == k, :);
    clusterStats(k).mean = mean(clusterData, 1);
    clusterStats(k).std = std(clusterData, 0, 1);
end

% Step 5: 可视化聚类结果
% 雷达图
figure;
for k = 1:numClusters
    subplot(1, numClusters, k);
    radarData = [clusterStats(k).mean, clusterStats(k).mean(1)]; % 闭合雷达图
    polarplot(linspace(0, 2*pi, numBins+1), radarData, '-o');
    title(['Cluster ', num2str(k)]);
end

% 箱线图
figure;
boxData = [];
groupLabels = [];
for k = 1:numClusters
    clusterData = features(idx == k, :); % 提取当前簇的数据
    boxData = [boxData; clusterData]; % 合并所有簇的数据
    groupLabels = [groupLabels; repmat(k, size(clusterData, 1), 1)]; % 添加组标签
end

% 将数据转置为列格式
boxData = boxData(:); % 展平为单列
groupLabels = repelem(groupLabels, numBins); % 每个特征点都带有簇标签

boxplot(boxData, groupLabels, 'Labels', arrayfun(@(x) ['Cluster ', num2str(x)], 1:numClusters, 'UniformOutput', false));
title('Feature Distribution across Clusters');

disp('聚类完成，统计结果如下：');
disp(clusterStats);
