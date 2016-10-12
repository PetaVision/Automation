------------
-- Params --
------------

local nbatch       = 8;
local learningRate = 0.001;

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

--------------------------------
-- TODO: Optional max pooling --
--------------------------------

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
   pv.addGroup(pvClassifier, layerName .. "ToEstimateError", {
            groupType       = "HyPerConn";
            channelCode     = -1;
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

   pv.addGroup(pvClassifier, layerName .. "ToCategoryEstimate", {
            groupType        = "CloneConn";
            channelCode      = 0;
            postLayerName    = "CategoryEstimate";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = layerName .. "ToEstimateError";
         }
      );
end

return pvClassifier;
