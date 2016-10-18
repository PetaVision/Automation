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
os.execute("cp "
      .. runName .. "/runs/trainclassify/*.pvp "
      .. runName .. "/weights");

-- Run test classifier
os.execute(cdPre .. mpiPreClass .. pathToBinary
           .. " -p params/" .. runName .. "_testclassify.params"
           .. " -t " .. numClassThreads .. mpiPostClass);

-- Run final analysis script
   --TODO
