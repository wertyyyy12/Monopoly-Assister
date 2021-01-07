setConfig;
[m_test, X_test, y_test] = loadImages(colorMode, imageWidth, imageHeight, test_set, labels);
[m, X, y] = loadImages(colorMode, imageWidth, imageHeight, training_set, labels);

lambda_vals = [0; 0.01; 0.03; 0.1; 0.3; 1; 3; 10];
pcts = zeros(size(lambda_vals));
  
for i = 1:length(lambda_vals)
  current_lambda = lambda_vals(i);
  fprintf('training with fmincg, lambda: %d... \n', current_lambda); %start training w/ given lambda
  initialParams = randInitialize(h_layers, input_units, h_units, h2_units, num_labels);

  costFunction = @(p) nnCost(X, y, p, num_labels, current_lambda, input_units, h_units, h2_units, 0, h_layers);
  [nn_params, cost] = fmincg(costFunction, initialParams, options);
  
  [J, grad, pct] = nnCost(X_test, y_test, nn_params, num_labels, current_lambda, input_units, h_units, h2_units, 1, h_layers);
  pcts(i) = pct;
end

disp([lambda_vals pcts]);
fprintf('     LAMBDA     ACCURACY \n');