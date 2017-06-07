function plot_histogram(activityPvp)
   [pvp, header] = readpvpfile(activityPvp);
   totalcount = 0;
   for i = 1:size(pvp)(1);
      if (size(pvp{i}.values)(1) > 0)
         if (size(pvp{i}.values)(2) > 0)
            count = hist(mod(pvp{i}.values(:,1), header.nf)+1, [1:header.nf]);
            totalcount += count;
         endif
      endif
   endfor
   bar(fliplr(sort(totalcount / (size(pvp)(1) * header.nx * header.ny))));
end
