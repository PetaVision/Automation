#!/usr/bin/lua


-- Run final analysis scripts
for index, layerName in pairs(layersToClassify) do
   os.execute("octave -q --eval \"disp(calc_sparsity('"
         .. runName .. "/sparse/test/" .. layerName .. ".pvp'));\" > "
         .. runName .. "/" .. layerName .. "_test_sparsity.txt");
   os.execute("octave -q --eval \"disp(calc_sparsity('"
         .. runName .. "/sparse/train/" .. layerName .. ".pvp'));\" > "
         .. runName .. "/" .. layerName .. "_train_sparsity.txt");
end

os.execute("octave -q --eval \"disp(calc_score('"
      .. runName .. "/runs/scoretrain/CategoryEstimate.pvp', '"
      .. runName .. "/groundtruth/train_gt.pvp'));\" > " .. runName .. "/deep_train_score.txt");
os.execute("octave -q --eval \"disp(calc_score('"
      .. runName .. "/runs/testclassify/CategoryEstimate.pvp', '"
      .. runName .. "/groundtruth/test_gt.pvp'));\" > " .. runName .. "/deep_test_score.txt");

if enableSimpleClassifier then
   os.execute("octave -q --eval \"disp(calc_score('"
         .. runName .. "/runs/scoretrain/SimpleCategoryEstimate.pvp', '"
         .. runName .. "/groundtruth/train_gt.pvp'));\" > " .. runName .. "/shallow_train_score.txt");
   os.execute("octave -q --eval \"disp(calc_score('"
         .. runName .. "/runs/testclassify/SimpleCategoryEstimate.pvp', '"
         .. runName .. "/groundtruth/test_gt.pvp'));\" > " .. runName .. "/shallow_test_score.txt");
end

os.execute("tail -n 1 " .. runName .."/*.txt");

