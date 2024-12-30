function activations = mactivation(trainedNet, img, aspect, slope, targetLayer)
    % Convert image to dlarray
    dlImg = dlarray(single(img), 'SSC');
    dlAspect = dlarray(single(aspect), 'SSC');
    dlSlope = dlarray(single(slope), 'SSC');
    
    % Convert DAGNetwork to layerGraph
    lgraph1 = layerGraph(trainedNet);

    % Find and remove classificationLayer if it exists
    layers = lgraph1.Layers;

    for i = 1:numel(layers)
        if isa(layers(i), 'nnet.cnn.layer.ClassificationOutputLayer')
            disp(['Found Classification Output Layer: ', layers(i).Name]);
            lgraph1 = removeLayers(lgraph1, layers(i).Name);
            disp('Removed Classification Output Layer');
            break;
        end
    end

    % Reconnect the network if necessary
    fcLayers = arrayfun(@(x) isa(x, 'nnet.cnn.layer.FullyConnectedLayer'), layers);
    if any(fcLayers)
        lastFCLayer = layers(find(fcLayers, 1, 'last')).Name;
        targetLayers = lgraph1.Connections.Destination;
        if ~ismember('softmax', targetLayers)
            lgraph1 = connectLayers(lgraph1, lastFCLayer, 'softmax');
        end
    end

    % Create dlnetwork object
    dlnet = dlnetwork(lgraph1);

    % Use dlfeval to ensure gradient tracking
    [grad, activations] = dlfeval(@computeGradients, dlnet, dlImg, dlAspect, dlSlope, targetLayer);

    % % Compute Grad-CAM
    % meanGrad = mean(grad, [1 2]);
    % gradCAMMap = sum(activations .* meanGrad, 3);
    % gradCAMMap = extractdata(gradCAMMap);
    % 
    % % Apply ReLU
    % gradCAMMap = max(gradCAMMap, 0);
    % 
    % % Normalize the Grad-CAM map
    % gradCAMMap = (gradCAMMap - min(gradCAMMap(:))) / (max(gradCAMMap(:)) - min(gradCAMMap(:)));
end

function [grad, activations] = computeGradients(dlnet, dlImg, dlAspect, dlSlope, targetLayer)
    % Forward pass through the network
    [scores, activations] = forward(dlnet, dlImg, dlAspect, dlSlope, 'Outputs', {'softmax', targetLayer});
    [sc, idx] = max(scores, [], 'all');
    classIdx = idx;
    % One-hot encoding for the target class
    oneHot = zeros([1 1 size(scores,1)], 'single');
    oneHot(1,1,classIdx) = 1;
    dlOneHot = dlarray(oneHot, 'SSC');

    % Backward pass to compute gradients
    grad = dlgradient(sum(scores .* dlOneHot, 'all'), activations);
end
