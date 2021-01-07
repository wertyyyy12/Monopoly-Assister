%%Refreshes the config variables to be used by program

%%---------------------- CONFIGURATION-----------------------
iterations = 50;
options = optimset('MaxIter', iterations);
lambda = 0.1;
h_layers = 1;
h_units = 2800;
h2_units = 1000;

colorMode = 'rgb';
imageWidth = 64;
imageHeight = 36;

labels = cellstr({'Illinois', 'Indiana', 'Kentucky'});
num_labels = length(labels);
training_set = './Training Set 4';
test_set = './Test Set 4';

%%--------------END MANUAL CONFIGURATION----------------------
multiplier = 0;
if startsWith(colorMode, 'rgb') == 1
  multiplier = 3;
endif

if startsWith(colorMode, 'bw') == 1
  multiplier = 1;
endif
input_units = imageWidth * imageHeight * multiplier;

%---------------------------------------------------------------