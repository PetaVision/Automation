
require("scripts.build_methods");

-- Pass the arguments -nosparse or -noclassify on the command line to skip
-- those runs. Pass both to only generate project files, but not run.

makeDirectories(runConfig);
backupScripts(runConfig, runParams);

-- Generate our classes.txt
if runParams.generateGroundTruth then
   makeClassesTxt(runConfig, runParams);
end

-- Add the PetaVision source directory to the package path
package.path = package.path .. ";"
               .. runConfig.pathToSource .. "/parameterWrapper/?.lua";

-- Import the PetaVision package 
pv = require "PVModule";

-- Generate our params files
local params = dofile(runParams.paramsFile);
local suffix;

------------------------------------
-- First run (learn a dictionary) --
------------------------------------

setNameAndLength(params, "learndictionary",
      runParams.displayPeriod * runParams.inputTrainFiles
      / runConfig.numSparseBatches * runParams.unsupervisedEpochs);

-- Set our input to training set and set display period 
for index, layerName in pairs(runParams.inputLayerNames) do
   params[layerName].displayPeriod = runParams.displayPeriod;
   params[layerName].inputPath     = runParams.inputTrainLists[index]; 
   params[layerName].batchMethod = "random";
end

for index, layerName in pairs(runParams.layersToClassify) do
   -- Don't write sparse code while learning
   params[layerName].initialWriteTime = -1;
   params[layerName].writeStep        = -1;
    
   -- Store the dimensions of the layers to classify for later
   runParams.layersToClassifyFeatures[layerName]  = params[layerName].nf;
   runParams.layersToClassifyXScale[layerName]    = params[layerName].nxScale;
   runParams.layersToClassifyYScale[layerName]    = params[layerName].nyScale;
end

local plasticConnIndex = 1;
for k, v in pairs(params) do
   if v.plasticityFlag == true then
      -- Write our plastic connections right at the end of the run
      runParams.plasticConns[plasticConnIndex] = k;
      v.initialWriteTime     = params.column.stopTime;
      v.writeStep            = params.column.stopTime;
      plasticConnIndex = plasticConnIndex + 1;
   end
end


-- Write the file and run it through PV with the dry run flag
sanitizeParamsFile(runConfig, params, "learndictionary");

-----------------------------------------------------
-- Second run (write sparse code for training set) --
-----------------------------------------------------
setNameAndLength(params, "writetrain",
   runParams.displayPeriod * runParams.inputTrainFiles / runConfig.numSparseBatches);

params.column.checkpointWrite = false;

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

for index, layerName in pairs(runParams.layersToClassify) do
   params[layerName].initialWriteTime = runParams.displayPeriod;
   params[layerName].writeStep        = runParams.displayPeriod;
   params[layerName].batchMethod = "byFile";
end

if runParams.generateGroundTruth then
   params["GroundTruth"] = {
         groupType         = "FilenameParsingGroundTruthLayer";
         phase             = params[ runParams.inputLayerNames[1] ].phase - 1;
         nxScale           = 1 / params.column.nx;
         nyScale           = 1 / params.column.ny;
         nf                = runParams.numCategories;
         writeStep         = runParams.displayPeriod;
         initialWriteTime  = runParams.displayPeriod;
         inputLayerName    = runParams.inputLayerNames[1];
         gtClassTrueValue  = 1;
         gtClassFalseValue = 0;
      };
end

sanitizeParamsFile(runConfig, params, "writetrain");

------------------------------------------------
-- Third run (write sparse code for test set) --
------------------------------------------------

setNameAndLength(params, "writetest",
   runParams.displayPeriod * runParams.inputTestFiles / runConfig.numSparseBatches);

for index, layerName in pairs(runParams.inputLayerNames) do
   params[layerName].inputPath = runParams.inputTestLists[index]; 
end

sanitizeParamsFile(runConfig, params, "writetest");

---------------------------------------------------------------
-- Fourth run (write maxpooled version of train sparse code) --
---------------------------------------------------------------

params = dofile("subnets/write_maxpool.lua");
setNameAndLength(params, "writemaxtrain",
   runParams.inputTrainFiles / runConfig.numClassBatches);

for index, layerName in pairs(runParams.layersToClassify) do
   if params[layerName] == nil then
      params[layerName .. "MaxPool"].inputPath = "sparse/train/"
                                 .. layerName .. ".pvp";
   else
      params[layerName].inputPath = "sparse/train/"
                                 .. layerName .. ".pvp";
   end
end

sanitizeParamsFile(runConfig, params, "writemaxtrain");

-------------------------------------------------------------
-- Fifth run (write maxpooled version of test sparse code) --
-------------------------------------------------------------

setNameAndLength(params, "writemaxtest",
   runParams.inputTestFiles / runConfig.numClassBatches);

for index, layerName in pairs(runParams.layersToClassify) do
   if params[layerName] == nil then
      params[layerName .. "MaxPool"].inputPath = "sparse/test/"
                                 .. layerName .. ".pvp";
   else
      params[layerName].inputPath = "sparse/test/"
                                 .. layerName .. ".pvp";
   end
end

sanitizeParamsFile(runConfig, params, "writemaxtest");

------------------------------------------------------------------
-- Sixth run (train classifier on sparse code of training set) --
------------------------------------------------------------------

params = dofile(runParams.classifier);

setNameAndLength(params, "trainclassify",
   runParams.inputTrainFiles / runConfig.numClassBatches * runParams.classifierEpochs);
params.column.checkpointWriteStepInterval = runParams.inputTrainFiles / runConfig.numClassBatches * 10;

for k, v in pairs(params) do
   if params[k].plasticityFlag == true then
      -- Write our plastic connections right at the end of the run
      params[k].initialWriteTime   = params.column.stopTime;
      params[k].writeStep          = params.column.stopTime;
      params[k].dWMaxDecayInterval = params.column.stopTime / runParams.numRateDecays;
   end
end

params.GroundTruth.inputPath = "groundtruth/train_gt.pvp";

for index, layerName in pairs(runParams.layersToClassify) do
   params[layerName].inputPath = "sparse/train/"
                                 .. layerName .. "MaxPool.pvp";
end

sanitizeParamsFile(runConfig, params, "trainclassify");

-----------------------------------------------------------
-- Seventh run (score classifier on sparse code of train set) --
-----------------------------------------------------------

setNameAndLength(params, "scoretrain", runParams.inputTrainFiles / runConfig.numClassBatches);
params.column.checkpointWrite     = false;

for k, v in pairs(params) do
   if v.plasticityFlag == true then
      v.plasticityFlag   = false;
      v.weightInitType   = "FileWeight";
      v.initWeightsFile  = "weights/" .. k .. ".pvp";
      v.initialWriteTime = -1;
      v.writeStep        = -1;
   end
end

for index, layerName in pairs(runParams.layersToClassify) do
   params[layerName].inputPath = "sparse/train/"
                                 .. layerName .. "MaxPool.pvp";
   params[layerName].batchMethod = "byFile";
end

for k,v in pairs(params) do
   if type(v) == "table" then
      if v["groupType"] == "DropoutLayer" then
         v["probability"] = 0;
      end
   end
end

if params.SimpleCategoryEstimate ~= nil then
   params.SimpleCategoryEstimate.writeStep        = 1;
   params.SimpleCategoryEstimate.initialWriteTime = 1;
end

params.CategoryEstimate.writeStep        = 1;
params.CategoryEstimate.initialWriteTime = 1;
params.GroundTruth.inputPath             = "groundtruth/train_gt.pvp";

sanitizeParamsFile(runConfig, params, "scoretrain");

-----------------------------------------------------------
-- Eighth run (run classifier on sparse code of test set) --
-----------------------------------------------------------

setNameAndLength(params, "testclassify", runParams.inputTestFiles / runConfig.numClassBatches);

for index, layerName in pairs(runParams.layersToClassify) do
   params[layerName].inputPath = "sparse/test/"
                                 .. layerName .. "MaxPool.pvp";
   params[layerName].batchMethod = "byFile";
end

if params.SimpleCategoryEstimate ~= nil then
   params.SimpleCategoryEstimate.writeStep        = 1;
   params.SimpleCategoryEstimate.initialWriteTime = 1;
end

params.CategoryEstimate.writeStep        = 1;
params.CategoryEstimate.initialWriteTime = 1;
params.GroundTruth.inputPath             = "groundtruth/test_gt.pvp";

sanitizeParamsFile(runConfig, params, "testclassify");

print("---------------------------------------\n");
print("  Finished generating " .. runConfig.runName .. "\n");
print("---------------------------------------\n");

local doSparse   = true;
local doClassify = true;
local doAnalysis = true;

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

      if k > 0 and v == "-noanalysis" then
         doAnalysis = false;
      end
   
      if k > 0 and v == "-phase" then
         foundPhase = true;
      end
   end
end

if doSparse then
   dofile("scripts/run_sparse.lua");
end

if doClassify then
   dofile("scripts/run_classify.lua");
end

if doAnalysis then
   dofile("scripts/run_analysis.lua");
end
