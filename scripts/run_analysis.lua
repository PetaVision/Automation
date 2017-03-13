#!/usr/bin/lua

if not singlePhase then
   -- Run final analysis scripts
   for index, layerName in pairs(runParams.layersToClassify) do
      os.execute("octave -q --eval \"disp(calc_sparsity('"
            .. runConfig.runName .. "/sparse/test/" .. layerName .. ".pvp'));\" > "
            .. runConfig.runName .. "/" .. layerName .. "_test_sparsity.txt");
      os.execute("octave -q --eval \"disp(calc_sparsity('"
            .. runConfig.runName .. "/sparse/train/" .. layerName .. ".pvp'));\" > "
            .. runConfig.runName .. "/" .. layerName .. "_train_sparsity.txt");
   end

   os.execute("octave -q --eval \"disp(calc_score('"
         .. runConfig.runName .. "/runs/scoretrain/CategoryEstimate.pvp', '"
         .. runConfig.runName .. "/groundtruth/train_gt.pvp'));\" > " .. runConfig.runName .. "/deep_train_score.txt");
   os.execute("octave -q --eval \"disp(calc_score('"
         .. runConfig.runName .. "/runs/testclassify/CategoryEstimate.pvp', '"
         .. runConfig.runName .. "/groundtruth/test_gt.pvp'));\" > " .. runConfig.runName .. "/deep_test_score.txt");

   if runParams.enableSimpleClassifier then
      os.execute("octave -q --eval \"disp(calc_score('"
            .. runConfig.runName .. "/runs/scoretrain/SimpleCategoryEstimate.pvp', '"
            .. runConfig.runName .. "/groundtruth/train_gt.pvp'));\" > " .. runConfig.runName .. "/shallow_train_score.txt");
      os.execute("octave -q --eval \"disp(calc_score('"
            .. runConfig.runName .. "/runs/testclassify/SimpleCategoryEstimate.pvp', '"
            .. runConfig.runName .. "/groundtruth/test_gt.pvp'));\" > " .. runConfig.runName .. "/shallow_test_score.txt");
   end

   os.execute("tail -n 1 " .. runConfig.runName .."/*.txt");
end
