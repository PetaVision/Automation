------------
-- Params --
------------

local maxPoolX       = 2;
local maxPoolY       = 2;
local nbatch         = numClassBatches;
local learningRate   = 0.001;
local hiddenFeatures = 512; 
local useGpu         = true;

-- TODO: Correct phase

-- This file requires the global variables:
--    numCategories,
--    columnWidth
--    columnHeight
--    layersToClassifyFeatures
--    layersToClassifyXScale
--    layersToClassifyYScale

------------
-- Column --
------------

local pvClassifier = {
   column = {
      groupType                  = "HyPerCol";
      nx                         = columnWidth;
      ny                         = columnHeight;
      startTime                  = 0;
      dt                         = 1; 
      progressInterval           = 100;
      randomSeed                 = 1234567890;
      nbatch                     = nbatch;
      checkpointWrite            = true;
      checkpointWriteTriggerMode = "step";
      deleteOlderCheckpoints     = false;
   }
};

------------
-- Layers --
------------

pv.addGroup(pvClassifier, "CategoryEstimate", {
         groupType        = "HyPerLayer";--"ANNLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         phase            = 3;
         writeStep        = -1;
         initialWriteTime = -1;
         -- VThresh          = 0;
         -- AMin             = 0;
         -- AMax             = 1;
         -- AShift           = 0;
      }
   );

pv.addGroup(pvClassifier, "Bias", {
         groupType        = "ConstantLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         initV            = "ConstantV";
         valueV           = 1.0;
         phase            = 3;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvClassifier, "HiddenError", {
         groupType        = "MaskLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = hiddenFeatures;
         phase            = 5;
         writeStep        = -1;
         initialWriteTime = -1;
         maskLayerName    = "Hidden";
         maskMethod       = "layer";
      }
   );

pv.addGroup(pvClassifier, "Hidden", {
         groupType        = "ANNLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = hiddenFeatures;
         phase            = 2;
         writeStep        = -1;
         initialWriteTime = -1;
         VThresh          = 0;
         AMin             = 0;
         AMax             = infinity;
         AShift           = 0;
      }
   );

pv.addGroup(pvClassifier, "EstimateError", {
         groupType        = "HyPerLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         phase            = 4;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvClassifier, "GroundTruth", {
         groupType        = "PvpLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         phase            = 0;
         displayPeriod    = 1;
         batchMethod      = "random";
         randomSeed       = 5;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

for index, layerName in pairs(layersToClassify) do
   pv.addGroup(pvClassifier, layerName, {
            groupType        = "PvpLayer";
            nxScale          = layersToClassifyXScale[layerName];
            nyScale          = layersToClassifyYScale[layerName];
            nf               = layersToClassifyFeatures[layerName];
            phase            = 0;
            displayPeriod    = 1;
            batchMethod      = "random";
            randomSeed       = 5;
            writeStep        = -1;
            initialWriteTime = -1;
            resetToStartOnLoop = false;
         }
      );

   pv.addGroup(pvClassifier, layerName .. "MaxPool", {
            groupType        = "HyPerLayer";
            nxScale          = maxPoolX / columnWidth;
            nyScale          = maxPoolY / columnHeight;
            nf               = layersToClassifyFeatures[layerName];
            phase            = 1;
            writeStep        = -1;
            initialWriteTime = -1;
            resetToStartOnLoop = false;
         }
      );
end

-----------------
-- Connections --
-----------------

pv.addGroup(pvClassifier, "GroundTruthToEstimateError", {
         groupType     = "IdentConn";
         channelCode   = 0;
         preLayerName  = "GroundTruth";
         postLayerName = "EstimateError";
      }
   );

pv.addGroup(pvClassifier, "CategoryEstimateToEstimateError", {
         groupType     = "IdentConn";
         channelCode   = 1;
         preLayerName  = "CategoryEstimate";
         postLayerName = "EstimateError";
      }
   );

pv.addGroup(pvClassifier, "BiasToEstimateError", {
         groupType       = "HyPerConn";
         channelCode     = -1;
         preLayerName    = "Bias";
         postLayerName   = "EstimateError";
         plasticityFlag  = true;
         nxp             = 1;
         nyp             = 1;
         nfp             = numCategories;
         dWMax           = learningRate / 10;
         weightInitType  = "UniformRandomWeight";
         wMinInit        = -0.0001;
         wMaxInit        = 0.0001;
         normalizeMethod = "normalizeL2";
         strength        = 1;
      }
   );

pv.addGroup(pvClassifier, "BiasToCategoryEstimate", {
         groupType        = "CloneConn";
         channelCode      = 0;
         preLayerName     = "Bias";
         postLayerName    = "CategoryEstimate";
         writeStep        = -1;
         initialWriteTime = -1;
         originalConnName = "BiasToEstimateError";
      }
   );

pv.addGroup(pvClassifier, "HiddenToEstimateError", {
         groupType       = "HyPerConn";
         channelCode     = -1;
         preLayerName    = "Hidden";
         postLayerName   = "EstimateError";
         plasticityFlag  = true;
         nxp             = 1;
         nyp             = 1;
         nfp             = numCategories;
         dWMax           = learningRate;
         weightInitType  = "UniformRandomWeight";
         wMinInit        = -0.01;
         wMaxInit        = 0.01;
         normalizeMethod = "normalizeL2";
         strength        = 1;
      }
   );

pv.addGroup(pvClassifier, "HiddenToCategoryEstimate", {
         groupType        = "CloneConn";
         channelCode      = 0;
         preLayerName     = "Hidden";
         postLayerName    = "CategoryEstimate";
         writeStep        = -1;
         initialWriteTime = -1;
         originalConnName = "HiddenToEstimateError";
      }
   );

pv.addGroup(pvClassifier, "EstimateErrorToHiddenError", {
         groupType        = "TransposeConn";
         channelCode      = 0;
         preLayerName     = "EstimateError";
         postLayerName    = "HiddenError";
         writeStep        = -1;
         initialWriteTime = -1;
         originalConnName = "HiddenToEstimateError";
      }
   );



for index, layerName in pairs(layersToClassify) do
   local maxPoolLayerName = layerName .. "MaxPool";

   pv.addGroup(pvClassifier, layerName .. "To" .. maxPoolLayerName, {
            groupType             = "PoolingConn";
            channelCode           = 0;
            preLayerName          = layerName;
            postLayerName         = maxPoolLayerName;
            pvpatchAccumulateType = "maxpooling";
            receiveGpu            = useGpu;
            writeStep             = -1;
            nxp                   = 1;
            nyp                   = 1;
            nfp                   = layersToClassifyFeatures[layerName];
         }
      );

   pv.addGroup(pvClassifier, maxPoolLayerName .. "ToHiddenError", {
            groupType       = "HyPerConn";
            channelCode     = -1;
            preLayerName    = maxPoolLayerName;
            postLayerName   = "HiddenError";
            plasticityFlag  = true;
            nxp             = 1;
            nyp             = 1;
            nfp             = hiddenFeatures;
            dWMax           = learningRate;
            weightInitType  = "UniformRandomWeight";
            wMinInit        = -0.01;
            wMaxInit        = 0.01;
            normalizeMethod = "normalizeL2";
            strength        = 1;
         }
      );

   pv.addGroup(pvClassifier, maxPoolLayerName .. "ToHidden", {
            groupType        = "CloneConn";
            channelCode      = 0;
            preLayerName     = maxPoolLayerName;
            postLayerName    = "Hidden";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = maxPoolLayerName .. "ToHiddenError";
         }
      );
end

return pvClassifier;
