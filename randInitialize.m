function initialParams = randInitialize(h_layers, input_units, h_units, h2_units, num_labels)
  if (h_layers == 1)
    epsilon_init = sqrt(6 / (input_units + h_units + num_labels));
    Theta1 = (rand(h_units, input_units + 1) .* (2*epsilon_init)) .- epsilon_init;
    Theta2 = (rand(num_labels, h_units + 1) .* (2*epsilon_init)) .- epsilon_init;
    initialParams = [Theta1(:); Theta2(:)];
  endif
  if (h_layers == 2)
    epsilon_init = sqrt(6 / (input_units + h_units + h2_units + num_labels));
    Theta1 = (rand(h_units, input_units + 1) .* (2*epsilon_init)) .- epsilon_init;
    Theta2 = (rand(h2_units, h_units + 1) .* (2*epsilon_init)) .- epsilon_init;
    Theta3 = (rand(num_labels, h2_units + 1) .* (2*epsilon_init)) .- epsilon_init;
    initialParams = [Theta1(:); Theta2(:); Theta3(:)];
  endif
end
