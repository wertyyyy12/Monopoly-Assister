% Load all of the images
function [m, X, y] = loadImages(colorMode, imageWidth, imageHeight, folder, labels)
  multiplier = 0;
  if startsWith(colorMode, 'rgb') == 1
    multiplier = 3;
  endif
  
  if startsWith(colorMode, 'bw') == 1
    multiplier = 1;
  endif
  
  images = {};
  imagefiles = dir(fullfile(folder, '*.jpg'));    
  nfiles = length(imagefiles); % Number of files found
  fprintf('Loading %d images from %s... ', nfiles, folder);
  X = zeros(nfiles, imageWidth * imageHeight * multiplier);
  m = nfiles;
  y = zeros(m, 1);
  for ii=1:nfiles
     baseFileName = imagefiles(ii).name;
     for l = labels
       if (startsWith(baseFileName, l{1}))
         y(ii) = find(ismember(labels, l{1}));
         break
       endif
     endfor
     fullFileName = fullfile(folder, baseFileName);
     currentimage = double(imread(fullFileName));
     images{ii} = currentimage;
  endfor
  
  j = 1;
  for i = images
    D = reshape(i{1}, imageHeight * multiplier, imageWidth);
    D = D(:) / 255;
    X(j, :) = D;
    j += 1;
  endfor
   
  fprintf('Loaded. \n');
end