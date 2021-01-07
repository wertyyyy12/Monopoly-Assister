setConfig;
[m_test, X_test, y_test] = loadImages(colorMode, imageWidth, imageHeight, test_set, labels);
[m, X, y] = loadImages(colorMode, imageWidth, imageHeight, training_set, labels);

lambda_vals = [0; 0.01; 0.03; 0.1; 0.3; 1; 3; 10];
iterations = [50; 150];
h_layers = [1; 2]; % number of hidden layers
h_layer_vals = [1000; 1200; 1400; 1600; 1800; 2000; 2200; 2400; 2600; 2800; 3000; 3200; 3400; 3600; 3800; 4000]; % number of units in each layer
h2_layer_vals = h_layer_vals(1:(length(h_layer_vals) - 11));

lambda_data = iteration_data = h_layer_data = h_layer_val_data = h2_layer_data = J_data = pct_data = zeros(size(h_layer_vals));



current_index = 1;
for a = 1:length(lambda_vals)
  for b = 1:length(iterations)
    for c = 1:length(h_layers)
      for d = 1:length(h_layer_vals)
        for e = 1:length(h2_layer_vals)
          current_lambda = lambda_vals(a);
          current_iteration = iterations(b);
          current_h_layer = h_layers(c);
          current_h_layer_val = h_layer_vals(d);
          current_h2_layer_val = h2_layer_vals(e);

          
          initialParams = randInitialize(current_h_layer, input_units, current_h_layer_val, current_h2_layer_val, num_labels);
          costFunction = @(p) nnCost(X, y, p, num_labels, current_lambda, input_units, current_h_layer_val, current_h2_layer_val, 0, current_h_layer);
          fprintf('Training with: lambda %d, iterations %d, # layers %d, h_units %d, h2_units %d \n', current_lambda, current_iteration, current_h_layer, current_h_layer_val, current_h2_layer_val);
          [nn_params, cost] = fmincg(costFunction, initialParams, options);
          
          [J, grad, pct] = nnCost(X_test, y_test, nn_params, num_labels, current_lambda, input_units, current_h_layer_val, current_h2_layer_val, 1, current_h_layer);
          lambda_data(current_index) = current_lambda;
          iteration_data(current_index) = current_iteration;
          h_layer_data(current_index) = current_h_layer;
          h_layer_val_data(current_index) = current_h_layer_val;
          h2_layer_data(current_index) = current_h2_layer_val;
          J_data(current_index) = J;
          pct_data(current_index) = pct;
          current_index += 1;
        end
      end
    end
  end
end

disp([lambda_data iteration_data h_layer_data h_layer_val_data h2_layer_data J_data pct_data]);
fprintf('     LAMBDA     ITERATIONS     # HIDDEN     HIDDEN 1     HIDDEN 2     J     ACCURACY \n');

