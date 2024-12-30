% from 冯鹏飞
% email：571428374@qq.com
% time:20241227
% Readme : The code is a visualization of the distribution states of 
% the identified results using the t-SNE approach
% 说明：该代码是使用t-SNE方法对识别后结果的分布状态进行可视化
clc;
clear; 
close all
%%
% Step 1: 加载图像数据集
imageFolder = 'non-landslide_improve_227'; % 修改为图像数据集所在路径 
% landslide_improve_227  landslide_improve_dem_227_aspect non-landslide_improve_dem_227_slope
imageFiles = dir(fullfile(imageFolder, '*.png')); % 修改为图像格式 png tiff
numImages = numel(imageFiles);
features = [];

filePath = '准确率结果（消融和对比）新.xlsx';
% 读取指定范围的数据
species = xlsread("准确率结果（消融和对比）新.xlsx",1,'D79:D278'); %D2:D78 D79:D278 F2:F78 F79:F278
% Step 2: 特征提取（灰度直方图）
numBins = 12; % 灰度直方图分箱数
for i = 1:numImages
    img = imread(fullfile(imageFolder, imageFiles(i).name));
    if size(img,3) > 1
        imgGray = rgb2gray(img);  % png需要
        histCounts = imhist(imgGray, numBins);
    else
        histCounts = imhist(img, numBins);
    end
    % histCounts = img(:)'; % 展平并转置为行向量
    % histCounts = histCounts / sum(histCounts); % 归一化
    features = [features; histCounts']; % 累积特征
end

% 确保 features 是浮点数组
features = double(features); % 或使用 single(features) 转为单精度浮点数
% Step 3: 使用 t-SNE 进行降维
rng default % for reproducibility
[reducedFeatures,loss] = tsne(features,'Algorithm','exact','Distance','euclidean');
rng default % for fair comparison
[reducedFeatures3,loss3] = tsne(features,'Algorithm','exact','Distance','euclidean', 'NumDimensions', 3);
fprintf('2-D embedding has loss %g, and 3-D embedding has loss %g.\n',loss,loss3)
% 可视化降维结果
figure;
gscatter(reducedFeatures(:,1),reducedFeatures(:,2),species)
str2title = imageFolder + " " + '2-D';  %2-D Embedding euclidean
str2title = strrep(str2title, '_', '-');
title(str2title)
% 设置图像大小为10cm*10cm，分辨率为300 DPI
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperSize', [10, 10]);
set(gcf, 'PaperPosition', [0, 0, 10, 10]);
fileName = sprintf(str2title);
% 构建完整的文件路径
output_folder = 'Singleresult';
output_file = fullfile(output_folder, fileName);
% 保存图像
print(gcf, output_file, '-dtiff', '-r300');

figure
v = double(categorical(species));
c = full(sparse(1:numel(v),v,ones(size(v)),numel(v),3));
scatter3(reducedFeatures3(:,1),reducedFeatures3(:,2),reducedFeatures3(:,3),15,c,'filled')
str3title = imageFolder + " " +'3-D';
str3title = strrep(str3title, '_', '-');
title(str3title)
% 设置图像大小为10cm*10cm，分辨率为300 DPI
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperSize', [10, 10]);
set(gcf, 'PaperPosition', [0, 0, 10, 10]);
fileName = sprintf(str3title);
% 构建完整的文件路径
output_folder = 'Singleresult';
output_file = fullfile(output_folder, fileName);
% 保存图像
print(gcf, output_file, '-dtiff', '-r300');

% Step 4: 基于降维后的特征进行分类（聚类）
% 使用 k-means 聚类
numClusters = 2; % 聚类数目，可根据实际需要调整
[idx, clusterCenters] = kmeans(reducedFeatures, numClusters);

[idx3, clusterCenters3] = kmeans(reducedFeatures3, numClusters);

% 可视化聚类结果
figure;
gscatter(reducedFeatures(:, 1), reducedFeatures(:, 2), idx, 'rgb', 'o', 10);
title('t-SNE with K-means Clustering');
legend('Cluster 1', 'Cluster 2', 'Cluster 2');
grid on;

figure;
v = double(categorical(idx3));
c = full(sparse(1:numel(v),v,ones(size(v)),numel(v),3));
scatter3(reducedFeatures3(:, 1), reducedFeatures3(:, 2), reducedFeatures3(:, 3), 15,c,'filled');
title('t-SNE with K-means Clustering');
% legend('Cluster 1', 'Cluster 2', 'Cluster 2');
grid on;

% Step 5: 统计各簇的特征均值和标准差
clusterStats = struct();
for k = 1:numClusters
    % clusterData = features(idx == k, :); % 获取当前簇的原始特征数据
    clusterData = features(species == k, :); % 获取当前簇的原始特征数据
    clusterStats(k).mean = mean(clusterData, 1); % 均值
    clusterStats(k).std = std(clusterData, 0, 1); % 标准差
end

% Step 6: 雷达图和箱线图对比
% 雷达图
figure;
for k = 1:numClusters
    subplot(1, numClusters, k);
    radarData = [clusterStats(k).mean]; % 闭合雷达图
    polarplot(linspace(0, 2*pi, numBins), radarData, '-o');
    title(['Cluster ', num2str(k)]);
end

feaSum = sum(features)'/size(features,1);

disp('聚类统计结果：');
disp(clusterStats);
