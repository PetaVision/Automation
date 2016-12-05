function plot_histogram(activityPvp)
   [pvp, header] = readpvpfile(activityPvp);
   totalcount = 0;
   for i = 1:size(pvp)(1);
      [count, center] = hist(mod(pvp{i}.values(:,1), header.nf)+1, [1:header.nf]);
      totalcount += count;
   end
   bar(fliplr(sort(totalcount / (size(pvp)(1) * header.nx * header.ny))));
end
