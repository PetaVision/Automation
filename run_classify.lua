#!/usr/bin/lua

local mpiPreClass  = "";
local mpiPostClass = "";
if numClassRows * numClassCols > 1 then
   mpiPreClass  = "mpiexec -np " .. (numClassRows * numClassCols) .. " ";
   mpiPostClass = " -rows " .. numClassRows
               .. " -columns " .. numClassCols;
end

local cdPre  = "cd " .. runName .. "; ";

-- Run train classifier
os.execute(cdPre .. mpiPreClass .. pathToBinary
           .. " -p params/" .. runName .. "_trainclassify.params"
           .. " -t " .. numClassThreads .. mpiPostClass);

-- Copy learned weights
for index, layerName in pairs(layersToClassify) do
   local fileName = layerName .. "ToEstimateError";
   print("Copying " .. fileName .. "\n");
   os.execute("cp "
         .. runName .. "/runs/trainclassify/" .. fileName .. ".pvp "
         .. runName .. "/weights");
end

-- Run test classifier
os.execute(cdPre .. mpiPreClass .. pathToBinary
           .. " -p params/" .. runName .. "_testclassify.params"
           .. " -t " .. numClassThreads .. mpiPostClass);

-- Run final analysis script
   --TODO
