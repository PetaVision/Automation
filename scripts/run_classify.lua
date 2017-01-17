#!/usr/bin/lua

if mpiBatchWidth == nil or mpiBatchWidth < 1 then
   print("Using default batchWidth of 1");
   mpiBatchWidth = 1;
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

-- Run train classifier
if not singlePhase or phaseToRun == 4 then
   os.execute(cdPre .. mpiPreClass .. pathToBinary
              .. " -p params/trainclassify.params"
              .. " -t " .. numClassThreads .. mpiPostClass);
   
   -- Copy learned weights
   if mpiBatchWidth > 1 then
      os.execute("cp "
            .. runName .. "/runs/trainclassify/batchsweep_00/*.pvp "
            .. runName .. "/weights; "
            .. "cd " .. runName .. "/weights; "
            .. "rename 's/_0\\.pvp/\\.pvp/' *.pvp"); -- Rename Weights_0.pvp to Weights.pvp
   else
      os.execute("cp "
            .. runName .. "/runs/trainclassify/*.pvp "
            .. runName .. "/weights");
   end
end

if not singlePhase or phaseToRun == 5 then
   -- Get score on train set
   os.execute(cdPre .. pathToBinary
              .. " -p params/scoretrain.params"
              .. " -t " .. numClassThreads);
   
   -- Get score on test set
   os.execute(cdPre .. pathToBinary
              .. " -p params/testclassify.params"
              .. " -t " .. numClassThreads);
end   
