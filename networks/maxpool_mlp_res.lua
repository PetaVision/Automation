------------
-- Params --
------------

local maxPoolX          = 4;
local maxPoolY          = 4;
local nbatch            = numClassBatches;
local learningRate      = 0.0000125; -- Divided by 8 for big run
local rateFactor        = 0.25; --1 / math.sqrt(maxPoolX * maxPoolY);
local hiddenFeatures    = 32; 
local useGpu            = true;
local weightStd         = 0.01;
local hiddenPatch       = 3;
local connectionType    = "MomentumConn";
local momentumType      = "simple";
local momentum          = 0.5;
local decayFactor       = 0.01; -- momentum decay is factor * learningRate for each conn
local biasValue         = 1.0;
local enableSoftmax     = true;
local inputDropout      = 15;
local hiddenDropout     = 40;
local normType          = "none";
local normStrength      = 1.0;
local normDW            = false; 
local sharedWeights     = true;
local debugWriteStep    = -1;
local allHiddenLayer    = true;
local allHiddenFeatures = 128;

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
      errorOnNotANumber          = true;
   }
};

------------
-- Layers --
------------

pv.addGroup(pvClassifier, "SoftmaxEstimate", {
         groupType         = "RescaleLayer";
         nxScale           = 1 / columnWidth;
         nyScale           = 1 / columnHeight;
         nf                = numCategories;
         phase             = 5;
         writeStep         = -1;
         initialWriteTime  = -1;
         rescaleMethod     = "softmax";
         originalLayerName = "CategoryEstimate";
         InitVType         = "ZeroV";
      }
   );

if not enableSoftmax then
   pvClassifier.SoftmaxEstimate.groupType = "CloneVLayer";
end

pv.addGroup(pvClassifier, "CategoryEstimate", {
         groupType        = "HyPerLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         phase            = 4;
         writeStep        = -1;
         initialWriteTime = -1;
         InitVType        = "ZeroV";
      }
   );

pv.addGroup(pvClassifier, "Bias", {
         groupType        = "ConstantLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         initV            = "ConstantV";
         valueV           = biasValue;
         phase            = 0;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvClassifier, "EstimateError", {
         groupType        = "HyPerLayer";
         nxScale          = 1 / columnWidth;
         nyScale          = 1 / columnHeight;
         nf               = numCategories;
         phase            = 6;
         writeStep        = -1;
         initialWriteTime = -1;
         InitVType        = "ZeroV";
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
         InitVType        = "ZeroV";
      }
   );

if allHiddenLayer then
   pv.addGroup(pvClassifier, "AllHiddenError", {
            groupType        = "MaskLayer";
            nxScale          = 1 / columnWidth;
            nyScale          = 1 / columnHeight;
            nf               = allHiddenFeatures;
            phase            = 7;
            writeStep        = -1;
            initialWriteTime = -1;
            maskLayerName    = "AllHidden";
            maskMethod       = "layer";
            InitVType        = "ZeroV";
         }
      );

   pv.addGroup(pvClassifier, "AllHidden", {
            groupType        = "DropoutLayer";
            nxScale          = 1 / columnWidth;
            nyScale          = 1 / columnHeight;
            nf               = allHiddenFeatures;
            phase            = 3;
            writeStep        = -1;
            initialWriteTime = -1;
            VThresh          = 0;
            AMin             = 0;
            AMax             = infinity;
            AShift           = 0;
            probability      = hiddenDropout;
            InitVType        = "ZeroV";
         }
      );
end

for index, layerName in pairs(layersToClassify) do
   pv.addGroup(pvClassifier, layerName, {
            groupType              = "PvpLayer";
            nxScale                = layersToClassifyXScale[layerName];
            nyScale                = layersToClassifyYScale[layerName];
            nf                     = layersToClassifyFeatures[layerName];
            phase                  = 0;
            displayPeriod          = 1;
            batchMethod            = "random";
            randomSeed             = 5;
            writeStep              = -1;
            initialWriteTime       = -1;
            resetToStartOnLoop     = false;
            normalizeLuminanceFlag = true;
            normalizeStdDev        = true;
            InitVType              = "ZeroV";
         }
      );

   pv.addGroup(pvClassifier, layerName .. "MaxPool", {
            groupType          = "DropoutLayer";
            nxScale            = maxPoolX / columnWidth;
            nyScale            = maxPoolY / columnHeight;
            nf                 = layersToClassifyFeatures[layerName];
            phase              = 1;
            writeStep          = -1;
            initialWriteTime   = -1;
            resetToStartOnLoop = false;
            VThresh            = -infinity;
            AMin               = -infinity;
            AMax               = infinity;
            AShift             = 0;
            probability        = inputDropout;
            InitVType          = "ZeroV";
         }
      );

   pv.addGroup(pvClassifier, layerName .. "HiddenError", {
            groupType        = "MaskLayer";
            nxScale          = maxPoolX / columnWidth;
            nyScale          = maxPoolY / columnHeight;
            nf               = hiddenFeatures;
            phase            = 8;
            writeStep        = -1;
            initialWriteTime = -1;
            maskLayerName    = layerName .. "Hidden";
            maskMethod       = "layer";
            InitVType        = "ZeroV";
         }
      );

   pv.addGroup(pvClassifier, layerName .. "Hidden", {
            groupType        = "DropoutLayer";
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
            probability      = hiddenDropout;
            InitVType        = "ZeroV";
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

pv.addGroup(pvClassifier, "SoftmaxEstimateToEstimateError", {
         groupType     = "IdentConn";
         channelCode   = 1;
         preLayerName  = "SoftmaxEstimate";
         postLayerName = "EstimateError";
      }
   );

if allHiddenLayer then
   pv.addGroup(pvClassifier, "AllHiddenToEstimateError", {
            groupType               = connectionType;
            momentumMethod          = momentumType;
            momentumTau             = momentum;
            momentumDecay           = decayFactor * learningRate;
            channelCode             = -1;
            preLayerName            = "AllHidden";
            postLayerName           = "EstimateError";
            plasticityFlag          = true;
            nxp                     = 1;
            nyp                     = 1;
            nfp                     = numCategories;
            dWMax                   = learningRate;
            weightInitType          = "GaussianRandomWeight";
            wGaussMean              = 0;
            wGaussStdev             = weightStd;
            normalizeMethod         = normType;
            strength                = normStrength;
            normalizeDw             = normDW;
            receiveGpu              = useGpu;
            initialWeightUpdateTime = 2;
            writeStep               = debugWriteStep;
            initialWriteTime        = 0;
         }
      );

   pv.addGroup(pvClassifier, "AllHiddenToCategoryEstimate", {
            groupType        = "CloneConn";
            channelCode      = 0;
            preLayerName     = "AllHidden";
            postLayerName    = "CategoryEstimate";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = "AllHiddenToEstimateError";
         }
      );

   pv.addGroup(pvClassifier, "EstimateErrorToAllHiddenError", {
            groupType        = "TransposeConn";
            channelCode      = 0;
            preLayerName     = "EstimateError";
            postLayerName    = "AllHiddenError";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = "AllHiddenToEstimateError";
         }
      );

   pv.addGroup(pvClassifier, "BiasToAllHiddenError", {
            groupType               = connectionType;
            momentumMethod          = momentumType;
            momentumTau             = momentum;
            momentumDecay           = decayFactor * learningRate / 2;
            channelCode             = -1;
            preLayerName            = "Bias";
            postLayerName           = "AllHiddenError";
            plasticityFlag          = true;
            nxp                     = 1;
            nyp                     = 1;
            nfp                     = allHiddenFeatures;
            dWMax                   = learningRate / 2;
            weightInitType          = "GaussianRandomWeight";
            wGaussMean              = 0;
            wGaussStdev             = 0;
            normalizeMethod         = normType;
            strength                = normStrength;
            normalizeDw             = normDW;
            initialWeightUpdateTime = 2;
         }
      );

   pv.addGroup(pvClassifier, "BiasToAllHidden", {
            groupType        = "CloneConn";
            channelCode      = 0;
            preLayerName     = "Bias";
            postLayerName    = "AllHidden";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = "BiasToAllHiddenError";
         }
      );
end

pv.addGroup(pvClassifier, "BiasToEstimateError", {
         groupType               = connectionType;
         momentumMethod          = momentumType;
         momentumTau             = momentum;
         momentumDecay           = decayFactor * learningRate / 2;
         channelCode             = -1;
         preLayerName            = "Bias";
         postLayerName           = "EstimateError";
         plasticityFlag          = true;
         nxp                     = 1;
         nyp                     = 1;
         nfp                     = numCategories;
         dWMax                   = learningRate / 2;
         weightInitType          = "GaussianRandomWeight";
         wGaussMean              = 0;
         wGaussStdev             = 0;
         normalizeMethod         = normType;
         strength                = normStrength;
         normalizeDw             = normDW;
         initialWeightUpdateTime = 2;
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

   if allHiddenLayer then
      pv.addGroup(pvClassifier, layerName .. "HiddenToAllHiddenError", {
               groupType               = connectionType;
               momentumMethod          = momentumType;
               momentumTau             = momentum;
               momentumDecay           = decayFactor * learningRate;
               channelCode             = -1;
               preLayerName            = layerName .. "Hidden";
               postLayerName           = "AllHiddenError";
               plasticityFlag          = true;
               nxp                     = 1;
               nyp                     = 1;
               nfp                     = allHiddenFeatures;
               dWMax                   = learningRate;
               weightInitType          = "GaussianRandomWeight";
               wGaussMean              = 0;
               wGaussStdev             = weightStd;
               normalizeMethod         = normType;
               strength                = normStrength;
               normalizeDw             = normDW;
               receiveGpu              = useGpu;
               initialWeightUpdateTime = 2;
               writeStep               = debugWriteStep;
               initialWriteTime        = 0;
            }
         );

      pv.addGroup(pvClassifier, layerName .. "HiddenToAllHidden", {
               groupType        = "CloneConn";
               channelCode      = 0;
               preLayerName     = layerName .. "Hidden";
               postLayerName    = "AllHidden";
               writeStep        = -1;
               initialWriteTime = -1;
               originalConnName = layerName .. "HiddenToAllHiddenError";
            }
         );

      pv.addGroup(pvClassifier, "AllHiddenErrorTo" .. layerName .. "HiddenError", {
               groupType        = "TransposeConn";
               channelCode      = 0;
               preLayerName     = "AllHiddenError";
               postLayerName    = layerName .. "HiddenError";
               writeStep        = -1;
               initialWriteTime = -1;
               originalConnName = layerName .. "HiddenToAllHiddenError";
            }
         );
   else
       pv.addGroup(pvClassifier, layerName .. "HiddenToEstimateError", {
               groupType               = connectionType;
               momentumMethod          = momentumType;
               momentumTau             = momentum;
               momentumDecay           = decayFactor * learningRate;
               channelCode             = -1;
               preLayerName            = layerName .. "Hidden";
               postLayerName           = "EstimateError";
               plasticityFlag          = true;
               nxp                     = 1;
               nyp                     = 1;
               nfp                     = numCategories;
               dWMax                   = learningRate;
               weightInitType          = "GaussianRandomWeight";
               wGaussMean              = 0;
               wGaussStdev             = weightStd;
               normalizeMethod         = normType;
               strength                = normStrength;
               normalizeDw             = normDW;
               receiveGpu              = useGpu;
               initialWeightUpdateTime = 2;
               writeStep               = debugWriteStep;
               initialWriteTime        = 0;
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
  
   end

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
            groupType               = connectionType;
            momentumMethod          = momentumType;
            momentumTau             = momentum;
            momentumDecay           = decayFactor * rateFactor * learningRate;
            channelCode             = -1;
            preLayerName            = maxPoolLayerName;
            postLayerName           = layerName .. "HiddenError";
            plasticityFlag          = true;
            nxp                     = hiddenPatch;
            nyp                     = hiddenPatch;
            nfp                     = hiddenFeatures;
            dWMax                   = rateFactor * learningRate;
            weightInitType          = "GaussianRandomWeight";
            wGaussMean              = 0;
            wGaussStdev             = weightStd;
            receiveGpu              = useGpu;
            normalizeMethod         = normType;
            strength                = normStrength;
            normalizeDw             = normDW;
            sharedWeights           = sharedWeights;
            initialWeightUpdateTime = 2;
            writeStep               = debugWriteStep;
            initialWriteTime        = 0;
         }
      );
   if not sharedWeights then
      pvClassifier[maxPoolLayerName .. "To" .. layerName .. "HiddenError"].receiveGpu = false;
   end

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

   pv.addGroup(pvClassifier, "BiasTo" .. layerName .. "HiddenError", {
            groupType               = connectionType;
            momentumMethod          = momentumType;
            momentumTau             = momentum;
            momentumDecay           = decayFactor * rateFactor * learningRate / 2;
            channelCode             = -1;
            preLayerName            = "Bias";
            postLayerName           = layerName .. "HiddenError";
            plasticityFlag          = true;
            nxp                     = maxPoolX;
            nyp                     = maxPoolY;
            nfp                     = hiddenFeatures;
            dWMax                   = rateFactor * learningRate / 2;
            weightInitType          = "GaussianRandomWeight";
            wGaussMean              = 0;
            wGaussStdev             = 0;
            normalizeMethod         = normType;
            strength                = normStrength;
            normalizeDw             = normDW;
            sharedWeights           = sharedWeights;
            initialWeightUpdateTime = 2;
         }
      );
   if not sharedWeights then
      pvClassifier["BiasTo" .. layerName .. "HiddenError"].receiveGpu = false;
   end


   pv.addGroup(pvClassifier, "BiasTo" .. layerName .. "Hidden", {
            groupType        = "CloneConn";
            channelCode      = 0;
            preLayerName     = "Bias";
            postLayerName    = layerName .. "Hidden";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = "BiasTo" .. layerName .. "HiddenError";
         }
      );
end

return pvClassifier;
