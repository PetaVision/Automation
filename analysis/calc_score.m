% Takes in the output of a classifier and the ground truth and scores it.
% Expects ground truth to be a one-hot 1x1xF vector.

function result = calc_score(estPvp, gtPvp)
   est   = readpvpfile(estPvp);
   gt    = readpvpfile(gtPvp);
   score = 0;
   total = size(gt)(1);
   for i=1:total
      [estVal, estInd] = max(est{i}.values);
      [gtVal, gtInd]   = max(gt{i}.values);
      if estInd == gtInd
         score += 1;
      end
   end
   result = score / total * 100.0;
   disp(["\t", num2str(score), "\t/ ", num2str(total), "\t(", num2str(result), "\%)"]);
end
