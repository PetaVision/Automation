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



-- Run inital training
if not singlePhase or phaseToRun == 0 then
   os.execute(cdPre .. mpiPreSparse .. pathToBinary
           .. " -p params/learndictionary.params"
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
end

if generateGroundTruth then
   -- If we have a FilenameParsingGroundTruthLayer,
   -- we can't split into rows / cols
   mpiPreSparse     = mpiPreClass;
   mpiPostSparse    = mpiPostClass;
   numSparseThreads = numClassThreads;
end

---- Run write train set
if not singlePhase or phaseToRun == 1 then
   os.execute(cdPre .. mpiPreSparse .. pathToBinary
              .. " -p params/writetrain.params"
              .. " -t " .. numSparseThreads .. mpiPostSparse);
   
   -- Move output files and rename ground truth if generated
   for index, layerName in pairs(layersToClassify) do
      if mpiBatchWidth > 1 then
         print("Merging batched files for " .. layerName .. "\n");
         os.execute("octave --eval \""
               .. "combinebatches('"
               .. runName .. "/runs/writetrain/', '"
               .. layerName .. "', "
               .. numSparseBatches .. "', '"
               .. "byFile', "
               .. mpiBatchWidth .. ", "
               .. inputTrainFiles .. ");\"; "
            .. "mv " .. layerName .. ".pvp "
                     .. runName .. "/runs/writetrain");
      end
      print("Moving " .. layerName .. ".pvp\n");
      os.execute("mv "
                 .. runName .. "/runs/writetrain/" .. layerName .. ".pvp "
                 .. runName .. "/sparse/train");
   end
   
   if mpiBatchWidth > 1 then
      os.execute("octave --eval \""
            .. "combinebatches('"
            .. runName .. "/runs/writetrain/', 'GroundTruth', "
            .. numSparseBatches .. "', '"
            .. "byFile', "
            .. mpiBatchWidth .. ", "
            .. inputTrainFiles .. ");\"");
      os.execute("mv GroundTruth.pvp "
                  .. runName .. "/runs/writetrain");
   end
   os.execute("mv "
                 .. runName .. "/runs/writetrain/GroundTruth.pvp "
                 .. runName .. "/groundtruth/train_gt.pvp");
end

-- Run write test set
if not singlePhase or phaseToRun == 2 then
   os.execute(cdPre .. mpiPreSparse .. pathToBinary
              .. " -p params/writetest.params"
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
               .. "byFile', "
               .. mpiBatchWidth .. ", "
               .. inputTestFiles .. ");\"; "
            .. "mv " .. layerName .. ".pvp "
                     .. runName .. "/runs/writetest");
      end
      print("Moving " .. layerName .. ".pvp\n");
      os.execute("mv "
                 .. runName .. "/runs/writetest/" .. layerName .. ".pvp "
                 .. runName .. "/sparse/test");
   end
   
   if mpiBatchWidth > 1 then
      os.execute("octave --eval \""
            .. "combinebatches('"
            .. runName .. "/runs/writetest/', 'GroundTruth', "
            .. numSparseBatches .. "', '"
            .. "byFile', "
            .. mpiBatchWidth .. ", "
            .. inputTestFiles .. ");\"; "
         .. "mv GroundTruth.pvp "
                  .. runName .. "/runs/writetest");
   end
   os.execute("mv "
                 .. runName .. "/runs/writetest/GroundTruth.pvp "
                 .. runName .. "/groundtruth/test_gt.pvp");
end

-- Write Maxpooled Test / Train
if not singlePhase or phaseToRun == 3 then
   os.execute(cdPre .. mpiPreSparse .. pathToBinary
              .. " -p params/writemaxtrain.params"
              .. " -t " .. numSparseThreads .. mpiPostSparse);
   
   -- Copy output files and rename ground truth if generated
   for index, layerName in pairs(layersToClassify) do
      if mpiBatchWidth > 1 then
         print("Merging batched files for " .. layerName .. "\n");
         os.execute("octave --eval \""
               .. "combinebatches('"
               .. runName .. "/runs/writemaxtrain/', '"
               .. layerName .. "MaxPool', "
               .. numSparseBatches .. "', '"
               .. "'byFile', "
               .. mpiBatchWidth .. ", "
               .. inputTestFiles .. ");\"; "
            .. "mv " .. layerName .. ".pvp "
                     .. runName .. "/runs/writemaxtrain");
      end
      print("Moving " .. layerName .. "MaxPool.pvp\n");
      os.execute("mv "
                 .. runName .. "/runs/writemaxtrain/" .. layerName .. "MaxPool.pvp "
                 .. runName .. "/sparse/test");
   end

   os.execute(cdPre .. mpiPreSparse .. pathToBinary
              .. " -p params/writemaxtest.params"
              .. " -t " .. numSparseThreads .. mpiPostSparse);
   
   -- Copy output files and rename ground truth if generated
   for index, layerName in pairs(layersToClassify) do
      if mpiBatchWidth > 1 then
         print("Merging batched files for " .. layerName .. "\n");
         os.execute("octave --eval \""
               .. "combinebatches('"
               .. runName .. "/runs/writemaxtest/', '"
               .. layerName .. "MaxPool', "
               .. numSparseBatches .. "', '"
               .. "'byFile', "
               .. mpiBatchWidth .. ", "
               .. inputTestFiles .. ");\"; "
            .. "mv " .. layerName .. ".pvp "
                     .. runName .. "/runs/writemaxtest");
      end
      print("Moving " .. layerName .. "MaxPool.pvp\n");
      os.execute("mv "
                 .. runName .. "/runs/writemaxtrain/" .. layerName .. "MaxPool.pvp "
                 .. runName .. "/sparse/test");
   end
end
