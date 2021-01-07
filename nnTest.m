setConfig;
[m, X, y] = loadImages(colorMode, imageWidth, imageHeight, test_set, labels);
nnCost(X, y, nn_params, num_labels, lambda, input_units, h_units, h2_units, 1, h_layers);