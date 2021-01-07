function starts = startsWith(s, prefix)
  starts = strncmp(s, prefix, length(prefix));
endfunction