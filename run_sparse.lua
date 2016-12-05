#!/usr/bin/lua

if mpiBatchWidth == nil or mpiBatchWidth < 1 then
   print("Using default batchWidth of 1");
   mpiBatchWidth = 1;
end

local mpiPreSparse  = "";
local mpiPostSparse = "";
local mpiSparseProcs      = numSparseRows * numSparseCols * mpiBatchWidth;
if mpiSparseProcs > 1 then
   mpiPreSparse  = "mpiexec -np " .. mpiSparseProcs .. " ";
   mpiPostSparse = " -rows " .. numSparseRows
                .. " -columns " .. numSparseCols
                .. " -batchwidth " .. mpiBatchWidth;
end

local mpiPreClass  = "";
local mpiPostClass = "";
local mpiClassProcs = numClassCols * numClassCols * mpiBatchWidth;
if mpiClassProcs > 1 then
   mpiPreClass  = "mpiexec -np " .. mpiClassProcs .. " ";
   mpiPostClass = " -rows " .. numClassRows
               .. " -columns " .. numClassCols
               .. " -batchwidth " .. mpiBatchWidth;
end

local cdPre  = "cd " .. runName .. "; ";


-- TODO: Allow resuming from any stage.

-- Run inital training
os.execute(cdPre .. mpiPreSparse .. pathToBinary
           .. " -p params/" .. runName .. "_learndictionary.params"
           .. " -t " .. numSparseThreads .. mpiPostSparse);

---- Copy dictionary to dictionary directory
for index, connName in pairs(plasticConns) do
   print("Copying " .. connName .. ".pvp\n");
   if mpiBatchWidth == 1 then
      os.execute("cp "
              .. runName .. "/runs/learndictionary/" .. connName .. ".pvp "
              .. runName .. "/dictionary");
   else
      os.execute("cp "
              .. runName .. "/runs/learndictionary/batchsweep_00/" .. connName .. ".pvp "
              .. runName .. "/dictionary/" .. connName .. ".pvp");
    end
end

if generateGroundTruth then
   -- If we have a FilenameParsingGroundTruthLayer,
   -- we can't split into rows / cols
   mpiPreSparse     = mpiPreClass;
   mpiPostSparse    = mpiPostClass;
   numSparseThreads = numClassThreads;
end

---- Run write train set
os.execute(cdPre .. mpiPreSparse .. pathToBinary
           .. " -p params/" .. runName .. "_writetrain.params"
           .. " -t " .. numSparseThreads .. mpiPostSparse);

-- Copy output files and rename ground truth if generated
for index, layerName in pairs(layersToClassify) do
   if mpiBatchWidth > 1 then
      print("Merging batched files for " .. layerName .. "\n");
      os.execute("octave --eval \""
            .. "combinebatches('"
            .. runName .. "/runs/writetrain/', '"
            .. layerName .. "', "
            .. numSparseBatches .. "', '"
            .. inputLayerBatchMethods[index] .. "', "
            .. mpiBatchWidth .. ", "
            .. inputTrainFiles .. ");\"; "
         .. "mv " .. layerName .. ".pvp "
                  .. runName .. "/runs/writetrain");
   end
   print("Copying " .. layerName .. ".pvp\n");
   os.execute("cp "
              .. runName .. "/runs/writetrain/" .. layerName .. ".pvp "
              .. runName .. "/sparse/train");
end

if mpiBatchWidth > 1 then
   os.execute("octave --eval \""
         .. "combinebatches('"
         .. runName .. "/runs/writetrain/', 'GroundTruth', "
         .. numSparseBatches .. "', '"
         .. inputLayerBatchMethods[1] .. "', "
         .. mpiBatchWidth .. ", "
         .. inputTrainFiles .. ");\"");
   os.execute("mv GroundTruth.pvp "
               .. runName .. "/runs/writetrain");
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
   if mpiBatchWidth > 1 then
      print("Merging batched files for " .. layerName .. "\n");
      os.execute("octave --eval \""
            .. "combinebatches('"
            .. runName .. "/runs/writetest/', '"
            .. layerName .. "', "
            .. numSparseBatches .. "', '"
            .. inputLayerBatchMethods[index] .. "', "
            .. mpiBatchWidth .. ", "
            .. inputTestFiles .. ");\"; "
         .. "mv " .. layerName .. ".pvp "
                  .. runName .. "/runs/writetest");
   end
   print("Copying " .. layerName .. ".pvp\n");
   os.execute("cp "
              .. runName .. "/runs/writetest/" .. layerName .. ".pvp "
              .. runName .. "/sparse/test");
end

if mpiBatchWidth > 1 then
   os.execute("octave --eval \""
         .. "combinebatches('"
         .. runName .. "/runs/writetest/', 'GroundTruth', "
         .. numSparseBatches .. "', '"
         .. inputLayerBatchMethods[1] .. "', "
         .. mpiBatchWidth .. ", "
         .. inputTestFiles .. ");\"; "
      .. "mv GroundTruth.pvp "
               .. runName .. "/runs/writetest");
end
os.execute("cp "
              .. runName .. "/runs/writetest/GroundTruth.pvp "
              .. runName .. "/groundtruth/test_gt.pvp");
