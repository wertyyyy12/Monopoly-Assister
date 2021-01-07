function X_norm = normalize(X)
  means = mean(X);
  stdDevs = std(X);
  
  X_norm = (X .- means) ./ stdDevs;
  
endfunction
