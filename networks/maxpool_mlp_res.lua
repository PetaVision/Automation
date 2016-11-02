------------
-- Params --
------------

local maxPoolX       = 4;
local maxPoolY       = 4;
local nbatch         = numClassBatches;
local learningRate   = 0.001;
local hiddenFeatures = 128; 
local useGpu         = true;

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

--pv.addGroup(pvClassifier, "SoftmaxEstimate", {
--         groupType        = "RescaleLayer";
--         nxScale          = 1 / columnWidth;
--         nyScale          = 1 / columnHeight;
--         nf               = numCategories;
--         phase            = 4;
--         writeStep        = -1;
--         initialWriteTime = -1;
--         rescaleMethod    = "softmax";
--         originalLayerName = "CategoryEstimate";
--      }
--   );

pv.addGroup(pvClassifier, "CategoryEstimate", {
         groupType        = "HyPerLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         phase            = 3;
         writeStep        = -1;
         initialWriteTime = -1;
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

pv.addGroup(pvClassifier, "EstimateError", {
         groupType        = "HyPerLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         phase            = 5;
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
            normalizeLuminanceFlag = true;
            normalizeStdDev = false;
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

   pv.addGroup(pvClassifier, layerName .. "HiddenError", {
            groupType        = "MaskLayer";
            nxScale          = maxPoolX / columnWidth;
            nyScale          = maxPoolY / columnHeight;
            nf               = hiddenFeatures;
            phase            = 6;
            writeStep        = -1;
            initialWriteTime = -1;
            maskLayerName    = layerName .. "Hidden";
            maskMethod       = "layer";
         }
      );

   pv.addGroup(pvClassifier, layerName .. "Hidden", {
            groupType        = "ANNLayer";
            nxScale          = maxPoolX / columnWidth;
            nyScale          = maxPoolY / columnHeight;
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

--pv.addGroup(pvClassifier, "SoftmaxEstimateToEstimateError", {
pv.addGroup(pvClassifier, "CategoryEstimateToEstimateError", {
         groupType     = "IdentConn";
         channelCode   = 1;
         preLayerName  = "CategoryEstimate"; --"SoftmaxEstimate";
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



for index, layerName in pairs(layersToClassify) do
   local maxPoolLayerName = layerName .. "MaxPool";

   pv.addGroup(pvClassifier, layerName .. "HiddenToEstimateError", {
            groupType       = "HyPerConn";
            channelCode     = -1;
            preLayerName    = layerName .. "Hidden";
            postLayerName   = "EstimateError";
            plasticityFlag  = true;
            nxp             = 1;
            nyp             = 1;
            nfp             = numCategories;
            dWMax           = learningRate;
            weightInitType  = "UniformRandomWeight";
            wMinInit        = -0.001;
            wMaxInit        = 0.001;
            normalizeMethod = "normalizeL2";
            strength        = 1;
            receiveGpu      = useGpu;
         }
      );

   pv.addGroup(pvClassifier, layerName .. "HiddenToCategoryEstimate", {
            groupType        = "CloneConn";
            channelCode      = 0;
            preLayerName     = layerName .. "Hidden";
            postLayerName    = "CategoryEstimate";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = layerName .. "HiddenToEstimateError";
         }
      );

   pv.addGroup(pvClassifier, "EstimateErrorTo" .. layerName .. "HiddenError", {
            groupType        = "TransposeConn";
            channelCode      = 0;
            preLayerName     = "EstimateError";
            postLayerName    = layerName .. "HiddenError";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = layerName .. "HiddenToEstimateError";
         }
      );

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

   pv.addGroup(pvClassifier, maxPoolLayerName .. "To" .. layerName .. "HiddenError", {
            groupType       = "HyPerConn";
            channelCode     = -1;
            preLayerName    = maxPoolLayerName;
            postLayerName   = layerName .. "HiddenError";
            plasticityFlag  = true;
            nxp             = 1;
            nyp             = 1;
            nfp             = hiddenFeatures;
            dWMax           = learningRate;
            weightInitType  = "UniformRandomWeight";
            wMinInit        = -0.001;
            wMaxInit        = 0.001;
            normalizeMethod = "normalizeL2";
            strength        = 1;
            receiveGpu      = useGpu;
         }
      );

   pv.addGroup(pvClassifier, maxPoolLayerName .. "To" .. layerName .. "Hidden", {
            groupType        = "CloneConn";
            channelCode      = 0;
            preLayerName     = maxPoolLayerName;
            postLayerName    = layerName .. "Hidden";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = maxPoolLayerName .. "To" .. layerName .. "HiddenError";
         }
      );
end

return pvClassifier;
