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
      .. runName .. "/groundtruth/train_gt.pvp'));\" > " .. runName .. "/train_score.txt");
os.execute("octave -q --eval \"disp(calc_score('"
      .. runName .. "/runs/testclassify/CategoryEstimate.pvp', '"
      .. runName .. "/groundtruth/test_gt.pvp'));\" > " .. runName .. "/test_score.txt");

if enableSimpleClassifier then
   os.execute("octave -q --eval \"disp(calc_score('"
         .. runName .. "/runs/scoretrain/SimpleCategoryEstimate.pvp', '"
         .. runName .. "/groundtruth/train_gt.pvp'));\" > " .. runName .. "/simple_train_score.txt");
   os.execute("octave -q --eval \"disp(calc_score('"
         .. runName .. "/runs/testclassify/SimpleCategoryEstimate.pvp', '"
         .. runName .. "/groundtruth/test_gt.pvp'));\" > " .. runName .. "/simple_test_score.txt");
end

os.execute("echo \'SPARSITY:\':");
os.execute("echo \'TRAIN: \'; cat " .. runName .. "/train_score.txt; "
        .. "echo \'TEST:  \'; cat " .. runName .. "/test_score.txt");

if enableSimpleClassifier then
   os.execute("echo \'SHALLOW CLASSIFICATION:\':");
   os.execute("echo \'TRAIN: \'; cat " .. runName .. "/simple_train_score.txt; "
           .. "echo \'TEST:  \'; cat " .. runName .. "/simple_test_score.txt");
end

os.execute("echo \'DEEP CLASSIFICATION:\':");
os.execute("echo \'TRAIN: \'; cat " .. runName .. "/train_score.txt; "
        .. "echo \'TEST:  \'; cat " .. runName .. "/test_score.txt");

