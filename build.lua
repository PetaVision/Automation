-- Pass the arguments -nosparse or -noclassify on the command line to skip
-- those runs. Pass both to only generate project files, but not run.

-- Create directories for each run and their results
local paramsDir = runName .. "/params/";
local runsDir   = runName .. "/runs/";
local luaDir    = runName .. "/lua/";
os.execute("mkdir -p " .. runName);
os.execute("mkdir -p " .. runName .. "/dictionary");
os.execute("mkdir -p " .. runName .. "/groundtruth");
os.execute("mkdir -p " .. runName .. "/sparse");
os.execute("mkdir -p " .. runName .. "/sparse/train");
os.execute("mkdir -p " .. runName .. "/sparse/test");
os.execute("mkdir -p " .. runName .. "/weights");
os.execute("mkdir -p " .. paramsDir);
os.execute("mkdir -p " .. luaDir);
os.execute("mkdir -p " .. runsDir);
os.execute("mkdir -p " .. runsDir .. "learndictionary");
os.execute("mkdir -p " .. runsDir .. "writetrain");
os.execute("mkdir -p " .. runsDir .. "writetest");
os.execute("mkdir -p " .. runsDir .. "trainclassify");
os.execute("mkdir -p " .. runsDir .. "testclassify");
os.execute("mkdir -p " .. runsDir .. "scoretrain");

-- Copy the lua files being used into the project for future reference
os.execute("cp " .. debug.getinfo(1).short_src .. " " .. luaDir);
os.execute("cp " .. paramsFile .. " " .. luaDir);
os.execute("cp " .. classifier .. " " .. luaDir);
os.execute("cp run_sparse.lua " .. runName .. "/.");
os.execute("cp run_classify.lua " .. runName .. "/.");

-- Generate our classes.txt
if generateGroundTruth then
   numCategories = 0;
   local classesFile = io.open(runsDir .. "writetrain/classes.txt", "w");
   io.output(classesFile);
   for index, class in pairs(classes) do
      io.write(class .. "\n");
      numCategories = numCategories + 1;
   end
   io.close(classesFile);
   if mpiBatchWidth > 1 then
      for mpiIndex = 0, mpiBatchWidth-1 do
         local batchPath = string.format("batchsweep_%02d/", mpiIndex);
         os.execute("mkdir -p " .. runsDir .. "writetrain/" .. batchPath);
         os.execute("mkdir -p " .. runsDir .. "writetest/" .. batchPath);
         os.execute("cp "
               .. runsDir .. "writetrain/classes.txt "
               .. runsDir .. "writetest/" .. batchPath);
         os.execute("cp "
               .. runsDir .. "writetrain/classes.txt "
               .. runsDir .. "writetrain/" .. batchPath);
      end
   else
      os.execute("cp "
            .. runsDir .. "writetrain/classes.txt "
            .. runsDir .. "writetest/classes.txt");
   end
end

-- Add the PetaVision source directory to the package path
package.path = package.path .. ";"
               .. pathToSource .. "/parameterWrapper/?.lua";

-- Import the PetaVision package 
pv = require "PVModule";

-- Generate our params files
local params = dofile(paramsFile);
local file;
local suffix;


------------------------------------
-- First run (learn a dictionary) --
------------------------------------

suffix = "learndictionary";

-- Set column params
params.column.outputPath =
      "runs/" .. suffix;
      
params.column.checkpointWriteDir =
      params.column.outputPath .. "/checkpoints";

params.column.printParamsFilename =
      runName .. "_" .. suffix .. ".params";

params.column.stopTime =
      displayPeriod * inputTrainFiles
    / params.column.nbatch * unsupervisedEpochs;

-- Set our input to training set and set display period 
for index, layerName in pairs(inputLayerNames) do
   params[layerName].displayPeriod = displayPeriod;
   params[layerName].inputPath     = inputTrainLists[index]; 
   params[layerName].batchMethod = "random";
end

for index, layerName in pairs(layersToClassify) do
   -- Write out our sparse code for analysis
   params[layerName].initialWriteTime = displayPeriod;
   params[layerName].writeStep        = displayPeriod;
    
   -- Store the dimensions of the layers to classify for later
   layersToClassifyFeatures[layerName]  = params[layerName].nf;
   layersToClassifyXScale[layerName]    = params[layerName].nxScale;
   layersToClassifyYScale[layerName]    = params[layerName].nyScale;
end

local plasticConnIndex = 1;
for k, v in pairs(params) do
   if v.plasticityFlag == true then
      -- Write our plastic connections right at the end of the run
      plasticConns[plasticConnIndex] = k;
      v.initialWriteTime     = params.column.stopTime;
      v.writeStep            = params.column.stopTime;
      plasticConnIndex = plasticConnIndex + 1;
   end
end

-- Write the file and run it through PV with the dry run flag
file = io.open(paramsDir .. params.column.printParamsFilename, "w");
io.output(file);
pv.printConsole(params);
io.close(file);


local command = 
      "cd " .. runName .. "; "
      .. pathToBinary .. " -p "
      .. "params/" .. params.column.printParamsFilename
      .. " -n; "
      .. "cd -; cp "
      .. runName .. "/runs/" .. suffix .. "/"
      .. params.column.printParamsFilename
      .. " " .. paramsDir;
os.execute(command);


-----------------------------------------------------
-- Second run (write sparse code for training set) --
-----------------------------------------------------

suffix = "writetrain";
params.column.checkpointWrite = false;
params.column.outputPath =
      "runs/" .. suffix;

params.column.checkpointWriteDir =
      params.column.outputPath .. "/checkpoints";

params.column.printParamsFilename =
      runName .. "_" .. suffix .. ".params";

params.column.stopTime =
      displayPeriod * inputTrainFiles / params.column.nbatch;

-- For every connection with plasticityFlag == true, disable
-- plasticity and use the weights learned in the first run.
for k, v in pairs(params) do
   if v.plasticityFlag == true then
      v.plasticityFlag   = false;
      v.weightInitType   = "FileWeight";
      v.initWeightsFile  = "dictionary/" .. k .. ".pvp";
      v.initialWriteTime = -1;
      v.writeStep        = -1;
   end
end

-- Random order shouldn't matter here, right?
--for index, layerName in pairs(inputLayerNames) do
--   params[layerName].batchMethod = "byFile";
--end

if generateGroundTruth then
   params["GroundTruth"] = {
         groupType         = "FilenameParsingGroundTruthLayer";
         phase             = params[ inputLayerNames[1] ].phase - 1;
         nxScale           = 1 / params.column.nx;
         nyScale           = 1 / params.column.ny;
         nf                = numCategories;
         writeStep         = displayPeriod;
         initialWriteTime  = displayPeriod;
         inputLayerName    = inputLayerNames[1];
         gtClassTrueValue  = 1;
         gtClassFalseValue = 0;
      };
end

-- Write the file and run it through PV with the dry run flag
file = io.open(paramsDir .. params.column.printParamsFilename, "w");
io.output(file);
pv.printConsole(params);
io.close(file);

command = 
      "cd " .. runName .. "; "
      .. pathToBinary .. " -p "
      .. "params/" .. params.column.printParamsFilename
      .. " -n; "
      .. "cd -; cp "
      .. runName .. "/runs/" .. suffix .. "/"
      .. params.column.printParamsFilename
      .. " " .. paramsDir;
os.execute(command);

------------------------------------------------
-- Third run (write sparse code for test set) --
------------------------------------------------

suffix = "writetest";

params.column.outputPath =
      "runs/" .. suffix;

params.column.checkpointWriteDir =
      params.column.outputPath .. "/checkpoints";

params.column.printParamsFilename =
      runName .. "_" .. suffix .. ".params";

params.column.stopTime =
      displayPeriod * inputTestFiles / params.column.nbatch;

for index, layerName in pairs(inputLayerNames) do
   params[layerName].inputPath = inputTestLists[index]; 
end

-- Write the file and run it through PV with the dry run flag
file = io.open(paramsDir .. params.column.printParamsFilename, "w");
io.output(file);
pv.printConsole(params);
io.close(file);

command = 
      "cd " .. runName .. "; "
      .. pathToBinary .. " -p "
      .. "params/" .. params.column.printParamsFilename
      .. " -n; "
      .. "cd -; cp "
      .. runName .. "/runs/" .. suffix .. "/"
      .. params.column.printParamsFilename
      .. " " .. paramsDir;
os.execute(command);

------------------------------------------------------------------
-- Fourth run (train classifier on sparse code of training set) --
------------------------------------------------------------------

params = dofile(classifier);

suffix = "trainclassify";

params.column.outputPath = 
      "runs/" .. suffix;

params.column.checkpointWriteDir =
      params.column.outputPath .. "/checkpoints";

params.column.printParamsFilename =
      runName .. "_" .. suffix .. ".params";

params.column.checkpointWriteStepInterval =
      inputTrainFiles / params.column.nbatch;

params.column.stopTime =
      classifierEpochs * params.column.checkpointWriteStepInterval;

params.column.checkpointWriteStepInterval = params.column.checkpointWriteStepInterval * 10;

for k, v in pairs(params) do
   if params[k].plasticityFlag == true then
      -- Write our plastic connections right at the end of the run
      params[k].initialWriteTime = params.column.stopTime;
      params[k].writeStep        = params.column.stopTime;
   end
end

params.GroundTruth.inputPath = "groundtruth/train_gt.pvp";

for index, layerName in pairs(layersToClassify) do
   params[layerName].inputPath = "sparse/train/"
                                 .. layerName .. ".pvp";
end

-- Write the file and run it through PV with the dry run flag
file = io.open(paramsDir .. params.column.printParamsFilename, "w");
io.output(file);
pv.printConsole(params);
io.close(file);

command = 
      "cd " .. runName .. "; "
      .. pathToBinary .. " -p "
      .. "params/" .. params.column.printParamsFilename
      .. " -n; "
      .. "cd -; cp "
      .. runName .. "/runs/" .. suffix .. "/"
      .. params.column.printParamsFilename
      .. " " .. paramsDir;
os.execute(command);


-----------------------------------------------------------
-- Fifth run (score classifier on sparse code of train set) --
-----------------------------------------------------------

suffix = "scoretrain";
params.column.outputPath          = "runs/" .. suffix;
params.column.checkpointWriteDir  = params.column.outputPath .. "/checkpoints";
params.column.printParamsFilename = runName .. "_" .. suffix .. ".params";
params.column.checkpointWrite     = false;
params.column.stopTime            = inputTrainFiles / params.column.nbatch;

for k, v in pairs(params) do
   if v.plasticityFlag == true then
      v.plasticityFlag   = false;
      v.weightInitType   = "FileWeight";
      v.initWeightsFile  = "weights/" .. k .. ".pvp";
      v.initialWriteTime = -1;
      v.writeStep        = -1;
   end
end

for index, layerName in pairs(layersToClassify) do
   params[layerName].inputPath = "sparse/train/"
                                 .. layerName .. ".pvp";
   params[layerName].batchMethod = "byFile";
   -- Hack to work with dropout, TODO: Find a cleaner way to do this
   params[layerName .. "Hidden"].probability = 0;
end


params.CategoryEstimate.writeStep        = 1;
params.CategoryEstimate.initialWriteTime = 1;
params.GroundTruth.inputPath             = "groundtruth/train_gt.pvp";

-- Write the file and run it through PV with the dry run flag
file = io.open(paramsDir .. params.column.printParamsFilename, "w");
io.output(file);
pv.printConsole(params);
io.close(file);

command = 
      "cd " .. runName .. "; "
      .. pathToBinary .. " -p "
      .. "params/" .. params.column.printParamsFilename
      .. " -n; "
      .. "cd -; cp "
      .. runName .. "/runs/" .. suffix .. "/"
      .. params.column.printParamsFilename
      .. " " .. paramsDir;
os.execute(command);

-----------------------------------------------------------
-- Sixth run (run classifier on sparse code of test set) --
-----------------------------------------------------------

suffix = "testclassify";
params.column.outputPath          = "runs/" .. suffix;
params.column.checkpointWriteDir  = params.column.outputPath .. "/checkpoints";
params.column.printParamsFilename = runName .. "_" .. suffix .. ".params";
params.column.checkpointWrite     = false;
params.column.stopTime            = inputTestFiles / params.column.nbatch;

for index, layerName in pairs(layersToClassify) do
   params[layerName].inputPath = "sparse/test/"
                                 .. layerName .. ".pvp";
   params[layerName].batchMethod = "byFile";
end

params.CategoryEstimate.writeStep        = 1;
params.CategoryEstimate.initialWriteTime = 1;
--params.CategoryEstimate.groupType        = "HyPerLayer"; -- Required if clamping
params.GroundTruth.inputPath             = "groundtruth/test_gt.pvp";

-- Write the file and run it through PV with the dry run flag
file = io.open(paramsDir .. params.column.printParamsFilename, "w");
io.output(file);
pv.printConsole(params);
io.close(file);

command = 
      "cd " .. runName .. "; "
      .. pathToBinary .. " -p "
      .. "params/" .. params.column.printParamsFilename
      .. " -n; "
      .. "cd -; cp "
      .. runName .. "/runs/" .. suffix .. "/"
      .. params.column.printParamsFilename
      .. " " .. paramsDir;
os.execute(command);


print("---------------------------------------\n");
print("  Finished generating " .. runName .. "\n");
print("---------------------------------------\n");

local doSparse   = true;
local doClassify = true;

singlePhase = false;
phaseToRun  = -1;

local foundPhase = false;

for k, v in pairs(arg) do

   if k > 0 and foundPhase then
      foundPhase = false;
      phaseToRun = tonumber(v);
      singlePhase = true;
   end

   if not foundPhase then
      if k > 0 and v == "-nosparse" then
         doSparse = false;
      end
   
      if k > 0 and v == "-noclassify" then
         doClassify = false;
      end
   
      if k > 0 and v == "-phase" then
         foundPhase = true;
      end
   end
end

if doSparse then
   dofile("run_sparse.lua");
end

if doClassify then
   dofile("run_classify.lua");
end
