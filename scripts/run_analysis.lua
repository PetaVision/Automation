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

os.execute("echo \'SPARSITY:\':");
os.execute("echo \'TRAIN: \'; cat " .. runName .. "/train_score.txt; "
        .. "echo \'TEST:  \'; cat " .. runName .. "/test_score.txt");

os.execute("echo \'CLASSIFICATION:\':");
os.execute("echo \'TRAIN: \'; cat " .. runName .. "/train_score.txt; "
        .. "echo \'TEST:  \'; cat " .. runName .. "/test_score.txt");

