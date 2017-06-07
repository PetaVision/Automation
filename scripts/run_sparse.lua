#!/usr/bin/lua

if runConfig.mpiBatchWidth == nil or runConfig.mpiBatchWidth < 1 then
   print("Using default batchWidth of 1");
   runConfig.mpiBatchWidth = 1;
end

local mpiPreSparse  = "";
local mpiPostSparse = "";
local mpiSparseProcs      = runConfig.numSparseRows * runConfig.numSparseCols * runConfig.mpiBatchWidth;
if mpiSparseProcs > 1 then
   mpiPreSparse  = "mpiexec -np " .. mpiSparseProcs .. " ";
   mpiPostSparse = " -rows " .. runConfig.numSparseRows
                .. " -columns " .. runConfig.numSparseCols
                .. " -batchwidth " .. runConfig.mpiBatchWidth;
end

local mpiPreClass  = "";
local mpiPostClass = "";
local mpiClassProcs = runConfig.numClassCols * runConfig.numClassCols * runConfig.mpiBatchWidth;
if mpiClassProcs > 1 then
   mpiPreClass  = "mpiexec -np " .. mpiClassProcs .. " ";
   mpiPostClass = " -rows " .. runConfig.numClassRows
               .. " -columns " .. runConfig.numClassCols
               .. " -batchwidth " .. runConfig.mpiBatchWidth;
end

local cdPre  = "cd " .. runConfig.runName .. "; ";



-- Run inital training
if not singlePhase or phaseToRun == 0 then
   os.execute(cdPre .. mpiPreSparse .. runConfig.pathToBinary
           .. " -p params/learndictionary.params"
           .. " -t " .. runConfig.numSparseThreads .. mpiPostSparse);

   ---- Copy dictionary to dictionary directory
   for index, connName in pairs(runParams.plasticConns) do
      print("Copying " .. connName .. ".pvp\n");
      os.execute("cp "
              .. runConfig.runName .. "/runs/learndictionary/" .. connName .. ".pvp "
              .. runConfig.runName .. "/dictionary");
   end
end

if runParams.generateGroundTruth then
   -- If we have a FilenameParsingGroundTruthLayer,
   -- we can't split into rows / cols
   mpiPreSparse     = mpiPreClass;
   mpiPostSparse    = mpiPostClass;
   numSparseThreads = numClassThreads;
end

---- Run write train set
if not singlePhase or phaseToRun == 1 then
   os.execute(cdPre .. mpiPreSparse .. runConfig.pathToBinary
              .. " -p params/writetrain.params"
              .. " -t " .. runConfig.numSparseThreads .. mpiPostSparse);
   
   -- Move output files and rename ground truth if generated
   for index, layerName in pairs(runParams.layersToClassify) do
      if runConfig.mpiBatchWidth > 1 then
         print("Merging batched files for " .. layerName .. "\n");
         os.execute("octave --eval \""
               .. "combinebatches('"
               .. runConfig.runName .. "/runs/writetrain/', '"
               .. layerName .. "', "
               .. runConfig.numSparseBatches .. "', '"
               .. "byFile', "
               .. runConfig.mpiBatchWidth .. ", "
               .. runParams.inputTrainFiles .. ");\"; "
            .. "mv " .. layerName .. ".pvp "
                     .. runConfig.runName .. "/runs/writetrain");
      end
      print("Moving " .. layerName .. ".pvp\n");
      os.execute("mv "
                 .. runConfig.runName .. "/runs/writetrain/" .. layerName .. ".pvp "
                 .. runConfig.runName .. "/sparse/train");
   end
   
   os.execute("mv "
                 .. runConfig.runName .. "/runs/writetrain/GroundTruth.pvp "
                 .. runConfig.runName .. "/groundtruth/train_gt.pvp");
end

-- Run write test set
if not singlePhase or phaseToRun == 2 then
   os.execute(cdPre .. mpiPreSparse .. runConfig.pathToBinary
              .. " -p params/writetest.params"
              .. " -t " .. runConfig.numSparseThreads .. mpiPostSparse);
   
   -- Copy output files and rename ground truth if generated
   for index, layerName in pairs(runParams.layersToClassify) do
      print("Moving " .. layerName .. ".pvp\n");
      os.execute("mv "
                 .. runConfig.runName .. "/runs/writetest/" .. layerName .. ".pvp "
                 .. runConfig.runName .. "/sparse/test");
   end
   
   os.execute("mv "
                 .. runConfig.runName .. "/runs/writetest/GroundTruth.pvp "
                 .. runConfig.runName .. "/groundtruth/test_gt.pvp");
end

-- Write Maxpooled Test / Train
if not singlePhase or phaseToRun == 3 then
   os.execute(cdPre .. mpiPreSparse .. runConfig.pathToBinary
              .. " -p params/writemaxtrain.params"
              .. " -t " .. runConfig.numSparseThreads .. mpiPostSparse);
   
   -- Copy output files and rename ground truth if generated
   for index, layerName in pairs(runParams.layersToClassify) do
      print("Moving " .. layerName .. "MaxPool.pvp\n");
      os.execute("mv "
                 .. runConfig.runName .. "/runs/writemaxtrain/" .. layerName .. "MaxPool.pvp "
                 .. runConfig.runName .. "/sparse/train");
   end

   os.execute(cdPre .. mpiPreSparse .. runConfig.pathToBinary
              .. " -p params/writemaxtest.params"
              .. " -t " .. runConfig.numSparseThreads .. mpiPostSparse);
   
   -- Copy output files and rename ground truth if generated
   for index, layerName in pairs(runParams.layersToClassify) do
      print("Moving " .. layerName .. "MaxPool.pvp\n");
      os.execute("mv "
                 .. runConfig.runName .. "/runs/writemaxtest/" .. layerName .. "MaxPool.pvp "
                 .. runConfig.runName .. "/sparse/test");
   end
end
