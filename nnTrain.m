
%%---------------------- CONFIGURATION-----------------------
setConfig;

%%---------------------END CONFIGURATION----------------------
multiplier = 0;
if startsWith(colorMode, 'rgb') == 1
  multiplier = 3;
endif

if startsWith(colorMode, 'bw') == 1
  multiplier = 1;
endif
input_units = imageWidth * imageHeight * multiplier;

%---------------------------------------------------------------
%initialize thetas randomly in [-epsilon_init, epsilon_init]
% epsilon_init = sqrt(6 / total units in NN)

initialParams = randInitialize(h_layers, input_units, h_units, h2_units, num_labels);
[m, X, y] = loadImages(colorMode, imageWidth, imageHeight, training_set, labels);
%-------------------------------------------------------------

% Create "short hand" for the cost function to be minimized

costFunction = @(p) nnCost(X, y, p, num_labels, lambda, input_units, h_units, h2_units, 0, h_layers);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
fprintf('training with fmincg... \n');
[nn_params, cost] = fmincg(costFunction, initialParams, options);
%cost
%-----------------------------------------------------------------

