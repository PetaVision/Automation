#!/usr/bin/lua

local cdPre  = "cd " .. runName .. "; ";

-- Run inital training
os.execute(cdPre .. pathToBinary
           .. " -p params/" .. runName .. "_learndictionary.params"
           .. " -t " .. numThreads);

-- Copy dictionary to dictionary directory
for index, connName in pairs(plasticConns) do
   print("Copying " .. connName .. ".pvp\n");
   os.execute("cp "
              .. runName .. "/runs/learndictionary/" .. connName .. ".pvp "
              .. runName .. "/dictionary");
end

-- Run write train set
os.execute(cdPre .. pathToBinary
           .. " -p params/" .. runName .. "_writetrain.params"
           .. " -t " .. numThreads);

-- Copy output files and rename ground truth if generated
for index, layerName in pairs(layersToClassify) do
   print("Copying " .. layerName .. ".pvp\n");
   os.execute("cp "
              .. runName .. "/runs/writetrain/" .. layerName .. ".pvp "
              .. runName .. "/sparse/train");
end

-- Run write test set
os.execute(cdPre .. pathToBinary
           .. " -p params/" .. runName .. "_writetest.params"
           .. " -t " .. numThreads);

-- Copy output files and rename ground truth if generated
for index, layerName in pairs(layersToClassify) do
   print("Copying " .. layerName .. ".pvp\n");
   os.execute("cp "
              .. runName .. "/runs/writetest/" .. layerName .. ".pvp "
              .. runName .. "/sparse/test");
end

-- Run train classifier
os.execute(cdPre .. pathToBinary
           .. " -p params/" .. runName .. "_trainclassify.params"
           .. " -t " .. numThreads);

-- Copy learned weights
for index, layerName in pairs(layersToClassify) do
   local fileName = layerName .. "ToEstimateError";
   print("Copying " .. fileName .. "\n");
   os.execute("cp "
         .. runName .. "/runs/trainclassify/" .. fileName .. ".pvp "
         .. runName .. "/weights");
end

-- Run test classifier
os.execute(cdPre .. pathToBinary
           .. " -p params/" .. runName .. "_testclassify.params"
           .. " -t " .. numThreads);

-- Run final analysis script
   --TODO
