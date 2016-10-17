------------
-- Params --
------------

local maxPoolX       = 4;
local maxPoolY       = 4;
local nbatch         = 8;
local learningRate   = 0.001;
local hiddenFeatures = 1024;

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
      deleteOlderCheckpoints     = true;
   }
};

------------
-- Layers --
------------

pv.addGroup(pvClassifier, "GroundTruth", {
         groupType        = "PvpLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         phase            = 0;
         displayPeriod    = 1;
         batchMethod      = "byFile";
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvClassifier, "CategoryEstimate", {
         groupType        = "HyPerLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         phase            = 2;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvClassifier, "EstimateError", {
         groupType        = "HyPerLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         phase            = 3;
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
            batchMethod      = "byFile";
            writeStep        = -1;
            initialWriteTime = -1;
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


for index, layerName in pairs(layersToClassify) do
   local maxPoolLayerName = layerName .. "MaxPool";

   pv.addGroup(pvClassifier, layerName .. "To" .. maxPoolLayerName, {
            groupType             = "PoolingConn";
            channelCode           = 0;
            preLayerName          = layerName;
            postLayerName         = maxPoolLayerName;
            pvpatchAccumulateType = "maxpooling";
            writeStep             = -1;
            nxp                   = 1;
            nyp                   = 1;
            nfp                   = layersToClassifyFeatures[layerName];
         }
      );

   pv.addGroup(pvClassifier, maxPoolLayerName .. "ToEstimateError", {
            groupType       = "HyPerConn";
            channelCode     = -1;
            preLayerName    = maxPoolLayerName;
            postLayerName   = "EstimateError";
            plasticityFlag  = true;
            nxp             = 1;
            nyp             = 1;
            nfp             = numCategories;
            dWMax           = learningRate;
            weightInitType  = "UniformRandomWeight";
            wMinInit        = -0.01;
            wMaxInit        = 0.01;
            normalizeMethod = "none";
         }
      );

   pv.addGroup(pvClassifier, maxPoolLayerName .. "ToCategoryEstimate", {
            groupType        = "CloneConn";
            channelCode      = 0;
            preLayerName     = maxPoolLayerName;
            postLayerName    = "CategoryEstimate";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = maxPoolLayerName .. "ToEstimateError";
         }
      );
end

return pvClassifier;
