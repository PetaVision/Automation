% Calculates the percent of non-zero elements over every frame in a PVP file.
% Assumes the input file is in sparse values format.

function result = calc_sparsity(pvpFileName)
   [data, header] = readpvpfile(pvpFileName);
   per_frame = 1.0 * header.nx * header.ny * header.nf;
   total = 0;
   for i = 1 : size(data)(1)
      total += (100.0 * size(data{i}.values)(1)) / per_frame;
   end
   result = total / size(data)(1);
end
