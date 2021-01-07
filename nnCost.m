function [J, grad, pct] = nnCost(X, y, nnParams, num_labels, lambda, input_units, h_units, h2_units, testing, h_layers)
  if (h_layers == 1)
    theta1pos = (h_units * (input_units + 1));
    theta2pos = theta1pos + (num_labels * (h_units + 1));
    Theta1 = reshape(nnParams(1:theta1pos), h_units, input_units + 1);
    Theta2 = reshape(nnParams((theta1pos + 1):theta2pos), num_labels, (h_units + 1));  

    m = rows(X);
    
    modifiedX = [ones(rows(X), 1) X];

    weightedSum2 = modifiedX * Theta1';
    activation2 = sigmoid(weightedSum2);
    % add bias
    modifiedA2 = [ones(rows(activation2), 1) activation2];
    weightedSum3 = modifiedA2 * Theta2';
    outputActivation = sigmoid(weightedSum3);

  %  modifiedA3 = [ones(rows(activation3), 1) activation3]; %add bias
  %  weightedSum4 = modifiedA3 * Theta3';
  %  outputActivation = sigmoid(weightedSum4);
    %-----------------------------------------------
    % convert the y vector into an NN friendly matrix
    convertedY = zeros(m, num_labels);
    for examplePosition = 1:m
      convertedY(examplePosition, y(examplePosition)) = 1;
    endfor
    %------------------------------------------------
    %calculate cost
    A = convertedY .* log(outputActivation);
    B = (1 - convertedY) .* log(1 - outputActivation);
    
    modifiedTheta1 = Theta1;
    modifiedTheta1(:, 1) = 0;

    modifiedTheta2 = Theta2;
    modifiedTheta2(:, 1) = 0;

    modifiedNNparams = [modifiedTheta1(:); modifiedTheta2(:)];


    J = -mean(sum(A + B, 2)) + ((lambda / (2*m) ) * sum(modifiedNNparams .^ 2)); %actually really elegant if you ask me


    
    %calculate gradients
    delta3 = outputActivation - convertedY; %for all training examples at once
    delta3 = delta3';


    partial = (Theta2' * delta3);
    partial(1, :) = [];
    delta2 = partial .* sigmoidGradient(weightedSum2');
  %
  %  partial2 = (Theta1' * delta2);
  %  partial2(1, :) = [];
  %  delta2 = partial2 .* sigmoidGradient(weightedSum2');


    Theta1_grad = ((delta2 * modifiedX) / m) + ((lambda / m) .* Theta1);
    Theta1_grad(:, 1) = Theta1_grad(:, 1) - ((lambda / m) .* Theta1)(:, 1);
    
    
    Theta2_grad = ((delta3 * modifiedA2) / m) + ((lambda / m) .* Theta2);
    Theta2_grad(:, 1) = Theta2_grad(:, 1) - ((lambda / m) .* Theta2)(:, 1);
    grad = [Theta1_grad(:); Theta2_grad(:)];
  endif  
  
  if (h_layers == 2)
    theta1pos = (h_units * (input_units + 1));
    theta2pos = theta1pos + (h2_units * (h_units + 1));
    theta3pos = theta2pos + (num_labels * (h2_units + 1));
    Theta1 = reshape(nnParams(1:theta1pos), h_units, input_units + 1);
    Theta2 = reshape(nnParams((theta1pos + 1):theta2pos), h2_units, (h_units + 1));  
    Theta3 = reshape(nnParams((theta2pos + 1):theta3pos), num_labels, (h2_units + 1));  

    m = rows(X);
    
    modifiedX = [ones(rows(X), 1) X];

    weightedSum2 = modifiedX * Theta1';
    activation2 = sigmoid(weightedSum2);
    % add bias
    modifiedA2 = [ones(rows(activation2), 1) activation2];
    weightedSum3 = modifiedA2 * Theta2';
    activation3 = sigmoid(weightedSum3);

    modifiedA3 = [ones(rows(activation3), 1) activation3]; %add bias
    weightedSum4 = modifiedA3 * Theta3';
    outputActivation = sigmoid(weightedSum4);
    %-----------------------------------------------
    % convert the y vector into an NN friendly matrix
    convertedY = zeros(m, num_labels);
    for examplePosition = 1:m
      convertedY(examplePosition, y(examplePosition)) = 1;
    endfor
    %------------------------------------------------
    %calculate cost
    A = convertedY .* log(outputActivation);
    B = (1 - convertedY) .* log(1 - outputActivation);
    
    modifiedTheta1 = Theta1;
    modifiedTheta1(:, 1) = 0;

    modifiedTheta2 = Theta2;
    modifiedTheta2(:, 1) = 0;

    modifiedTheta3 = Theta3;
    modifiedTheta3(:, 1) = 0;
    
    modifiedNNparams = [modifiedTheta1(:); modifiedTheta2(:); modifiedTheta3(:)];


    J = -mean(sum(A + B, 2)) + ((lambda / (2*m) ) * sum(modifiedNNparams .^ 2)); %actually really elegant if you ask me


    
    %calculate gradients
    delta4 = outputActivation - convertedY; %for all training examples at once
    delta4 = delta4';


    partial = (Theta3' * delta4);
    partial(1, :) = [];
    delta3 = partial .* sigmoidGradient(weightedSum3');
  %
    partial2 = (Theta2' * delta3);
    partial2(1, :) = [];
    delta2 = partial2 .* sigmoidGradient(weightedSum2');


    Theta1_grad = ((delta2 * modifiedX) / m) + ((lambda / m) .* Theta1);
    Theta1_grad(:, 1) = Theta1_grad(:, 1) - ((lambda / m) .* Theta1)(:, 1);
    
    
    Theta2_grad = ((delta3 * modifiedA2) / m) + ((lambda / m) .* Theta2);
    Theta2_grad(:, 1) = Theta2_grad(:, 1) - ((lambda / m) .* Theta2)(:, 1);
    
    
    Theta3_grad = ((delta4 * modifiedA3) / m) + ((lambda / m) .* Theta3);
    Theta3_grad(:, 1) = Theta3_grad(:, 1) - ((lambda / m) .* Theta3)(:, 1);
    
    
    grad = [Theta1_grad(:); Theta2_grad(:); Theta3_grad(:)];
  endif
  
  if (testing == 1)
      [max, ind] = max(outputActivation, [], 2);
      correctPredictions = sum(ind == y);
      pctCorrect = (correctPredictions / m) * 100;
      pct = pctCorrect;
      
      fprintf('%d correct out of %d. Thats %d percent. \n', correctPredictions, m, pctCorrect);
      
      predictedOne = sum(ind == 1);
      predictedTwo = sum(ind == 2);
      predictedThree = sum(ind == 3);
      fprintf('%d predictions were 1, %d were 2, %d were 3. \n', predictedOne, predictedTwo, predictedThree);
  endif
end
