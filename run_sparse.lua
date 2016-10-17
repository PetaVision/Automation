#!/usr/bin/lua

local mpiPreSparse  = "";
local mpiPostSparse = "";
if numSparseRows * numSparseCols > 1 then
   mpiPreSparse  = "mpiexec -np " .. (numSparseRows * numSparseCols) .. " ";
   mpiPostSparse = " -rows " .. numSparseRows
                .. " -columns " .. numSparseCols;
end

local mpiPreClass  = "";
local mpiPostClass = "";
if numClassRows * numClassCols > 1 then
   mpiPreClass  = "mpiexec -np " .. (numClassRows * numClassCols) .. " ";
   mpiPostClass = " -rows " .. numClassRows
               .. " -columns " .. numClassCols;
end

local cdPre  = "cd " .. runName .. "; ";

-- Run inital training
os.execute(cdPre .. mpiPreSparse .. pathToBinary
           .. " -p params/" .. runName .. "_learndictionary.params"
           .. " -t " .. numSparseThreads .. mpiPostSparse);

-- Copy dictionary to dictionary directory
for index, connName in pairs(plasticConns) do
   print("Copying " .. connName .. ".pvp\n");
   os.execute("cp "
              .. runName .. "/runs/learndictionary/" .. connName .. ".pvp "
              .. runName .. "/dictionary");
end

if generateGroundTruth then
   -- If we have a FilenameParsingGroundTruthLayer,
   -- we can't split into rows / cols
   mpiPreSparse     = mpiPreClass;
   mpiPostSparse    = mpiPostClass;
   numSparseThreads = numClassThreads;
end

-- Run write train set
os.execute(cdPre .. mpiPreSparse .. pathToBinary
           .. " -p params/" .. runName .. "_writetrain.params"
           .. " -t " .. numSparseThreads .. mpiPostSparse);

-- Copy output files and rename ground truth if generated
for index, layerName in pairs(layersToClassify) do
   print("Copying " .. layerName .. ".pvp\n");
   os.execute("cp "
              .. runName .. "/runs/writetrain/" .. layerName .. ".pvp "
              .. runName .. "/sparse/train");
end
os.execute("cp "
              .. runName .. "/runs/writetrain/GroundTruth.pvp "
              .. runName .. "/groundtruth/train_gt.pvp");

-- Run write test set
os.execute(cdPre .. mpiPreSparse .. pathToBinary
           .. " -p params/" .. runName .. "_writetest.params"
           .. " -t " .. numSparseThreads .. mpiPostSparse);

-- Copy output files and rename ground truth if generated
for index, layerName in pairs(layersToClassify) do
   print("Copying " .. layerName .. ".pvp\n");
   os.execute("cp "
              .. runName .. "/runs/writetest/" .. layerName .. ".pvp "
              .. runName .. "/sparse/test");
end
os.execute("cp "
              .. runName .. "/runs/writetest/GroundTruth.pvp "
              .. runName .. "/groundtruth/test_gt.pvp");
