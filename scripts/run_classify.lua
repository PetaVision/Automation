#!/usr/bin/lua

if runConfig.mpiBatchWidth == nil or runConfig.mpiBatchWidth < 1 then
   print("Using default batchWidth of 1");
   runConfig.mpiBatchWidth = 1;
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

-- Run train classifier
if not singlePhase or phaseToRun == 4 then
   os.execute(cdPre .. mpiPreClass .. runConfig.pathToBinary
              .. " -p params/trainclassify.params"
              .. " -t " .. runConfig.numClassThreads .. mpiPostClass);
   
   -- Copy learned weights
   os.execute("cp "
         .. runConfig.runName .. "/runs/trainclassify/*.pvp "
         .. runConfig.runName .. "/weights");
end

if not singlePhase or phaseToRun == 5 then
   -- Get score on train set
   os.execute(cdPre .. runConfig.pathToBinary
              .. " -p params/scoretrain.params"
              .. " -t " .. runConfig.numClassThreads);
   
   -- Get score on test set
   os.execute(cdPre .. runConfig.pathToBinary
              .. " -p params/testclassify.params"
              .. " -t " .. runConfig.numClassThreads);
end   
