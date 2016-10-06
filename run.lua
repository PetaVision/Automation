#!/usr/bin/lua

-- Run inital training
os.execute("cd " .. runName);
os.execute(pathToBinary
           .. " - p params/" .. runName .. "_learndictionary.params"
           .. " -t " .. numThreads);

-- Copy dictionary to dictionary directory
for index, connName in pairs(plasticConns) do
   print("Copying " .. connName .. ".pvp\n");
   os.execute("cp runs/learndictionary/" .. connName .. ".pvp /dictionary");
end

-- Run write train set
os.execute(pathToBinary
           .. " - p params/" .. runName .. "_writetrain.params"
           .. " -t " .. numThreads);

-- Copy output files and rename ground truth if generated
for index, layerName in pairs(layersToClassify) do
   print("Copying " .. layerName .. ".pvp\n");
   os.execute("cp runs/writetrain/" .. layerName .. ".pvp /sparse/train");
end

-- Run write test set
os.execute(pathToBinary
           .. " - p params/" .. runName .. "_writetest.params"
           .. " -t " .. numThreads);

-- Copy output files and rename ground truth if generated
for index, layerName in pairs(layersToClassify) do
   print("Copying " .. layerName .. ".pvp\n");
   os.execute("cp runs/writetest/" .. layerName .. ".pvp /sparse/test");
end

-- Run train classifier
os.execute(pathToBinary
           .. " - p params/" .. runName .. "_trainclassify.params"
           .. " -t " .. numThreads);

-- Copy learned weights
for index, layerName in pairs(layersToClassify) do
   local fileName = layerName .. "ToEstimateError.pvp";
   print("Copying " .. fileName .. "\n");
   os.execute("cp runs/trainclassify/" .. fileName .. " /weights");
end

-- Run test classifier
os.execute(pathToBinary
           .. " - p params/" .. runName .. "_testclassify.params"
           .. " -t " .. numThreads);

-- Run final analysis script
