%from 冯鹏飞
%email：571428374@qq.com & fpf0103@163.com
%time:20241127
% Readme: This code is for the interpretability of DFF and
% its visualization for each operator layer
% of the full stage of a multimodal deep learning model
% 说明：该代码是用于多模态深度学习模型全阶段各算子层DFF的可解释性及其可视化
clc
clear
%%
% 定义参数
% 读取并预处理输入图像
img = imread('..\227\landslide_improve_227\zj073.png');
% 读取并预处理坡向信息
aspect = imread('..\aspect\landslide_improve_dem_227_aspect\zj073.tiff');
if size(aspect, 3) ~= 1
    aspect = reshape(aspect, [227, 227, 1]);
end
% 读取并预处理坡度信息
slope = imread('..\slope\landslide_improve_dem_227_slope\zj073.tiff');
if size(slope, 3) ~= 1
    slope = reshape(slope, [227, 227, 1]);
end

n_components = 2;                        % 分解的概念数量
top_k = 2;                               % 每个概念显示的标签数量
labels_file_path = 'landslide2_clsidx_to_labels.txt';% 标签文件路径

% 加载模型
net = load('Final_AC_Swish_trained_model.mat').net; % 可以选择其他模型
targetlayer = 'aspc_concat';
% 调用函数并显示结果
[result_image,visualization,masks] = visualize_image_dff(net, targetlayer, img, aspect, slope, n_components, top_k, labels_file_path);

% 显示结果
figure;
imshow(result_image,'border','tight','initialmagnification','fit');
axis normal;
 
%%
function [result,visualization,masks] = visualize_image_dff(model, targetlayer, img_url, aspect, slope, n_components, top_k, labels_file_path)
    % Step 1: Load and preprocess image
    img = img_url;
    img_resized = imresize(img, [227, 227]); % Resize as per the CNN input requirements
    aspect_resized = imresize(aspect, [227, 227]); % Resize as per the CNN input requirements
    slope_resized = imresize(slope, [227, 227]); % Resize as per the CNN input requirements
    dlImg = dlarray(single(img_resized), 'SSC');
    dlAspect = dlarray(single(aspect_resized), 'SSC');
    dlSlope = dlarray(single(slope_resized), 'SSC');

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
    % features = activations(net, img_resized, layer, 'OutputAs', 'channels');
    [scores, features] = forward(dlnet, dlImg, dlAspect, dlSlope, 'Outputs', {'softmax', layer});
    % [ascores, afeatures] = activations(dlnet, dlImg, dlAspect, dlSlope, 'Outputs', {'softmax', layer});
    features = single(extractdata(features));

    % Step 3: Apply Non-negative Matrix Factorization (NMF)
    [h, w, c] = size(features);
    features_flattened = reshape(features, [h * w, c]);
    features_flattened = features_flattened';
    % 计算每行的最小值
    min_values = min(features_flattened, [], 2);
    % 将每行减去各自的最小值
    features_normalized = features_flattened - min_values;
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
        heatmaps{i} = imresize(heatmap, [size(img, 1), size(img, 2)]);
    end

    % Step 5: Overlay heatmaps on the original image
    % 调用 show_factorization_on_image 来生成解释叠加图
    [visualization,masks] = show_factorization_on_image(img, heatmaps, 0.7);    
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

function [result_image,masks] = show_factorization_on_image(img, explanations, image_weight, concept_labels_topk)
    % img: 输入的RGB图像
    % explanations: 分解区域的解释信息 (num_components x height x width)
    % colors: 每个区域对应的颜色
    % image_weight: 叠加图像的透明度权重
    % concept_labels: 每个区域的标签，用于图例显示

    % 参数设置
    n_components = size(explanations,2);
    [height, width] = size(explanations{1});
    
    colors = hsv(n_components); % 使用HSV颜色映射
    % % 创建图像窗口
    % figure;
    % hold on;
    % for i = 1:n_components
    %     % 绘制颜色矩形
    %     rectangle('Position', [0, n_components-i, 1, 1], 'FaceColor', colors(i, :), 'EdgeColor', 'none');
    %     % 添加颜色标签
    %     text(1.2, n_components-i+0.5, sprintf('Color %d', i), 'FontSize', 12, 'VerticalAlignment', 'middle');
    % end
    % 
    % % 调整坐标轴范围和样式
    % xlim([0, 2]); % 显示范围，保证矩形和标签都在可见范围内
    % ylim([0, n_components]); % Y轴范围覆盖所有颜色块
    % axis off; % 关闭坐标轴显示
    % title('HSV Colors with Labels'); % 添加标题
    % hold off;
    
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

    for i = 1:n_components
        % 创建一个与 img 大小相同的全零矩阵，并设置颜色
        mask = zeros(size(img, 1), size(img, 2), 3);
        mask(:, :, 1) = colors(i, 1);
        mask(:, :, 2) = colors(i, 2);
        mask(:, :, 3) = colors(i, 3);

        % 获取第 i 个解释（explanation）并将不属于当前 concept 的部分置零
        explanation = explanations{i};
        explanation(concept_per_pixel ~= i-1) = 0;

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

    % threshold = graythresh(mask);  % 自动计算阈值
    % binary_img = imbinarize(mask, threshold);
    % figure;
    % imshow(binary_img,'border','tight','initialmagnification','fit');
    % axis normal;
    
    % 将掩膜和原始图像进行融合
    img_double = double(img) / 255;
    result_image = img_double * image_weight + mask * (1 - image_weight);
end

