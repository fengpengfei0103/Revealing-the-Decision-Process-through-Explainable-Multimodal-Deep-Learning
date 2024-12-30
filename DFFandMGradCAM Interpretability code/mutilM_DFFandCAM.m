%from 冯鹏飞
%email：571428374@qq.com & fpf0103@163.com
%time:20241202
% Readme: This code is for the interpretability and its visualization of 
% the combination of DFF and CAM for each operator layer 
% in the full stage of a multimodal deep learning model
% 说明：该代码是用于多模态深度学习模型全阶段各算子层DFF与CAM结合的可解释性及其可视化
clc
clear all
%%
% 定义参数
n_components = 2;                        % 分解的概念数量
top_k = 2;                               % 每个概念显示的标签数量
labels_file_path = 'landslide2_clsidx_to_labels.txt';% 标签文件路径

% 加载模型
net = load('Final_AC_Swish_trained_model.mat').net; % 可以选择其他模型
analyzeNetwork(net)
%%
% 读取并预处理输入图像
lab = 'fyb947';  % 你可以根据需要改变这个值，读取不同的图片
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
% score
% targetLayer = 'conv_E3';
% targetLayer = 'cwconv11_E2';
% targetLayer = 'aspc_concat';
% targetLayer = 'aspc_relu_4';
% targetLayer = 'aspc_relu_3';
% targetLayer = 'aspc_relu_2';
% targetLayer = 'aspc_relu_1';
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
% 调用函数并显示结果
[result_image,visualization,markSpace1, heatmaps1,mask] = visualize_image_dff(net, targetLayer, img, aspect, slope, n_components, top_k, labels_file_path);

% 生成MM-Grad-CAM
mmGradCAMMap = mgradCAM(net, img, aspect, slope, targetLayer);
% 将Grad-CAM叠加到原始图像上
mmGradCAMMap = imresize(mmGradCAMMap, [size(img,1) size(img,2)], 'bilinear');

% 创建文件夹“结果”
output_folder = lab;
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
% 文件名
filename = '得分.txt';

% 检查文件是否存在
if ~isfile([output_folder,'\',filename])
    % 如果文件不存在，创建一个空文件
    fileID = fopen([output_folder,'\',filename], 'w');
    fclose(fileID);
end

for i=1:size(markSpace1,2)
    colorScore{i} =  nnz(mmGradCAMMap.*markSpace1{i})/nnz(markSpace1{i});
    text = sprintf('影响分数 %d: %.4f', i, colorScore{i});
    disp(text)
    % 打开文件以追加模式写入
    fileID = fopen([output_folder,'\',filename], 'a');

    % 写入文本并换行
    fprintf(fileID, '%s%s\n',targetLayer, text);

    % 关闭文件
    fclose(fileID);
end

% 显示结果
figure;
imshow(img,'border','tight','initialmagnification','fit');
axis normal;
hold on;
imagesc(mask,'AlphaData',0.6);
colormap jet
hold off;
% 设置图像大小为5cm*5cm，分辨率为300 DPI
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperSize', [5, 5]);
set(gcf, 'PaperPosition', [0, 0, 5, 5]);
fileName = sprintf('%s_%s_%s.tif', lab,targetLayer, 'DFFhalf');
% 构建完整的文件路径
output_file = fullfile(output_folder, fileName);
% 保存图像
print(gcf, output_file, '-dtiff', '-r300');
disp(['图像已保存到文件夹 "', output_folder, '" 中，文件名为 "', sprintf('%s_%s_%s.tif', lab,targetLayer, 'DFFhalf'), '"']);

figure;
imshow(img,'border','tight','initialmagnification','fit');
axis normal;
hold on;
imagesc(mmGradCAMMap,'AlphaData',0.5);
colormap jet
hold off;
% 设置图像大小为5cm*5cm，分辨率为300 DPI
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperSize', [5, 5]);
set(gcf, 'PaperPosition', [0, 0, 5, 5]);
fileName = sprintf('%s_%s_%s.tif', lab,targetLayer, 'mmGradCAMMap');
% 构建完整的文件路径
output_file = fullfile(output_folder, fileName);
% 保存图像
print(gcf, output_file, '-dtiff', '-r300');
disp(['图像已保存到文件夹 "', output_folder, '" 中，文件名为 "', sprintf('%s_%s_%s.tif', lab,targetLayer, 'DFFhalf'), '"']);

%%
function [result,visualization,markSpace,features_normalized,mask] = visualize_image_dff(model, targetlayer, img_url, aspect, slope, n_components, top_k, labels_file_path)
    % Step 1: Load and preprocess image
    img = img_url;
    img_resized = imresize(img, [227, 227]); % Resize as per the CNN input requirements
    aspect_resized = imresize(aspect, [227, 227]); % Resize as per the CNN input requirements
    slope_resized = imresize(slope, [227, 227]); % Resize as per the CNN input requirements
    dlImg = dlarray(single(img_resized), 'SSC');
    dlAspect = dlarray(single(aspect_resized), 'SSC');
    dlSlope = dlarray(single(slope_resized), 'SSC');

    % 添加批量维度
    dlImg = cat(4, dlImg, []);           % 添加批量维度，变为 'SSCB'
    dlAspect = cat(4, dlAspect, []);
    dlSlope = cat(4, dlSlope, []);

    % Step 2: Extract features using pretrained CNN (e.g., VGG19)
    % Convert DAGNetwork to layerGraph
    lgraph = layerGraph(model);
    % Find and remove classificationLayer if it exists
    layers = lgraph.Layers;
    for i = 1:numel(layers)
        if isa(layers(i), 'nnet.cnn.layer.ClassificationOutputLayer')
            disp(['Found Classification Output Layer: ', layers(i).Name]);
            lgraph = removeLayers(lgraph, layers(i).Name);
            disp('Removed Classification Output Layer');
            break;
        end
    end
    % Reconnect the network if necessary
    fcLayers = arrayfun(@(x) isa(x, 'nnet.cnn.layer.FullyConnectedLayer'), layers);
    if any(fcLayers)
        lastFCLayer = layers(find(fcLayers, 1, 'last')).Name;
        targetLayers = lgraph.Connections.Destination;
        if ~ismember('softmax', targetLayers)
            lgraph = connectLayers(lgraph, lastFCLayer, 'softmax');
        end
    end
    % Create dlnetwork object
    dlnet = dlnetwork(lgraph);
    layer = targetlayer; % Specify deep layer for feature extraction
    [scores, features] = forward(dlnet, dlImg, dlAspect, dlSlope, 'Outputs', {'softmax', layer});
    
    features = single(extractdata(features));

    % Step 3: Apply Non-negative Matrix Factorization (NMF)
    [h, w, c] = size(features);
    features_flattened = reshape(features, [h * w, c]);
    features_flattened = features_flattened';
    % 计算每行的最小值
    min_values = min(features_flattened, [], 2);
    % 将每行减去各自的最小值
    features_normalized = features_flattened - min_values;
    rng(0); % 设置随机数生成器种子
    [W, H] = nnmf(features_normalized, n_components); % Perform NMF

    concepts  = W + min_values;
   
    % Step 4: Generate heatmaps and concept labels
    % 将 H 重塑为 (n_components, batch_size, h, w) 维度
    % explanations = reshape(H, n_components, h, w);
    heatmaps = cell(1, n_components);
    % concept_labels_topk = create_labels(concept_outputs, top_k, labels_file_path);
    for i = 1:n_components
        heatmap = reshape(H(i, :), [h, w]);
        % 将 img 减去其最小值
        heatmap = heatmap - min(heatmap(:));
        % 将 img 除以 (1e-7 + 最大值)
        heatmap = heatmap / (1e-7 + max(heatmap(:)));
        heatmaps{i} = imresize(heatmap, [size(img, 1), size(img, 2)], 'bilinear');
    end

    % Step 5: Overlay heatmaps on the original image
    % 调用 show_factorization_on_image 来生成解释叠加图
    % visualization = show_factorization_on_image(img, heatmaps, 0.5, colors, concept_labels_topk);
    [visualization,markSpace,mask] = show_factorization_on_image(img, heatmaps, 0.7);

    % Concatenate original and visualization images for display
    % figure;
    % imshow(img,'border','tight','initialmagnification','fit');
    % axis normal;
    % figure;
    % imshow(visualization,'border','tight','initialmagnification','fit');
    % axis normal;
    

    result = cat(2, img, uint8(visualization*255));
end

function concept_labels_topk = create_labels(concept_scores, top_k, labels_file_path)
    % Load labels from file
    labels = load_labels(labels_file_path);
    
    % Get top_k concept labels for each component
    concept_labels_topk = cell(1, size(concept_scores, 1));
    for i = 1:size(concept_scores, 1)
        [sorted_scores, sorted_indices] = sort(concept_scores(i, :), 'descend');
        labels_str = "";
        for j = 1:top_k
            score = sorted_scores(j);
            label = labels{sorted_indices(j)};
            labels_str = strcat(labels_str, sprintf("%s:%.2f\n", label, score));
        end
        concept_labels_topk{i} = labels_str;
    end
end

function labels = load_labels(labels_file_path)
    % Load labels from a text file
    fid = fopen(labels_file_path, 'r');
    labels = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    labels = labels{1};
end

function [result_image,markSpace,mask] = show_factorization_on_image(img, explanations, image_weight, concept_labels_topk)
    % img: 输入的RGB图像
    % explanations: 分解区域的解释信息 (num_components x height x width)
    % colors: 每个区域对应的颜色
    % image_weight: 叠加图像的透明度权重
    % concept_labels: 每个区域的标签，用于图例显示

    % 参数设置
    n_components = size(explanations,2);
    [height, width] = size(explanations{1});
    
    colors = hsv(n_components); % 使用HSV颜色映射
    
    
    % 获取每个像素属于的概念区域
    % [~, concept_per_pixel] = max(explanations, [], 1);
    % 将 cell 数组中的矩阵沿第三维度连接成一个三维数组
    combined_matrix = cat(3, explanations{:});
    % 在第三维度上找到每个元素的最大值所在的索引
    [~, concept_per_pixel] = max(combined_matrix, [], 3);
    % 将索引值减 1，使得 H 中的值表示相应的矩阵最大
    concept_per_pixel = concept_per_pixel - 1;


    % 创建掩膜并叠加在原图像上
    masks = cell(1, n_components);  % 用于存储生成的 mask
    markSpace = cell(1, n_components);

    for i = 1:n_components
        % 创建一个与 img 大小相同的全零矩阵，并设置颜色
        mask = zeros(size(img, 1), size(img, 2), 3);
        mask(:, :, 1) = colors(i, 1);
        mask(:, :, 2) = colors(i, 2);
        mask(:, :, 3) = colors(i, 3);

        % 获取第 i 个解释（explanation）并将不属于当前 concept 的部分置零
        explanation = explanations{i};
        explanation(concept_per_pixel ~= i-1) = 0;
        markSpace{i} = explanation;

        % 将 mask 转换到 HSV 空间，并设置亮度通道
        mask = uint8(mask * 255);
        mask = rgb2hsv(mask);
        mask(:, :, 3) = uint8(255 * explanation);

        % 将 mask 转换回 RGB 空间，并归一化
        mask = hsv2rgb(mask);
        mask = double(mask) / 255;

        % 将生成的 mask 添加到 cell 数组中
        masks{i} = mask;
        % figure;
        % imshow(mask)

    end
    
    % 将 masks 中的每个 3D 矩阵转换为双精度，并逐元素求和
    mask = sum(cat(4, masks{:}), 4);
    
    % 将掩膜和原始图像进行融合
    img_double = double(img) / 255;
    result_image = img_double * image_weight + mask * (1 - image_weight);
    % figure;
    % imshow(img_double,'border','tight','initialmagnification','fit');
    % axis normal;
    % hold on;
    % imagesc(mask,'AlphaData',0.5);
    % colormap jet
    % hold off;
    % % 显示图例（在图像旁边显示标签）
    % if ~isempty(concept_labels)
    %     figure;
    %     imshow(result_image);
    %     hold on;
    %     for i = 1:n_components
    %         % 绘制每个区域的图例
    %         rectangle('Position', [size(result_image, 2) + 10, i * 30, 20, 20], ...
    %             'FaceColor', colors(i, :), 'EdgeColor', 'none');
    %         text(size(result_image, 2) + 40, i * 30 + 10, concept_labels{i}, 'FontSize', 12);
    %     end
    %     hold off;
    % else
    %     imshow(result_image);
    % end
end

