
------------
-- Column --
------------

local pvClassifier = {
   column = {
      groupType                  = "HyPerCol";
      nx                         = runParams.columnWidth;
      ny                         = runParams.columnHeight;
      startTime                  = 0;
      dt                         = 1; 
      progressInterval           = 100;
      randomSeed                 = 1234567890;
      nbatch                     = runConfig.numClassBatches;
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
         nxScale           = 1 / runParams.columnWidth;
         nyScale           = 1 / runParams.columnHeight;
         nf                = runParams.numCategories;
         phase             = 5;
         writeStep         = -1;
         initialWriteTime  = -1;
         rescaleMethod     = "softmax";
         originalLayerName = "CategoryEstimate";
         InitVType         = "ZeroV";
      }
   );

pv.addGroup(pvClassifier, "CategoryEstimate", {
         groupType        = "HyPerLayer";
         nxScale          = 1 / runParams.columnWidth;
         nyScale          = 1 / runParams.columnHeight;
         nf               = runParams.numCategories;
         phase            = 4;
         writeStep        = -1;
         initialWriteTime = -1;
         InitVType        = "ZeroV";
      }
   );

if runParams.enableSimpleClassifier then
   pv.addGroup(pvClassifier, "SimpleSoftmaxEstimate", {
            groupType         = "RescaleLayer";
            nxScale           = 1 / runParams.columnWidth;
            nyScale           = 1 / runParams.columnHeight;
            nf                = runParams.numCategories;
            phase             = 5;
            writeStep         = -1;
            initialWriteTime  = -1;
            rescaleMethod     = "softmax";
            originalLayerName = "SimpleCategoryEstimate";
            InitVType         = "ZeroV";
         }
      );

   pv.addGroup(pvClassifier, "SimpleCategoryEstimate", {
            groupType        = "HyPerLayer";
            nxScale          = 1 / runParams.columnWidth;
            nyScale          = 1 / runParams.columnHeight;
            nf               = runParams.numCategories;
            phase            = 4;
            writeStep        = -1;
            initialWriteTime = -1;
            InitVType        = "ZeroV";
         }
      );
   pv.addGroup(pvClassifier, "SimpleEstimateError", {
            groupType        = "HyPerLayer";
            nxScale          = 1 / runParams.columnWidth;
            nyScale          = 1 / runParams.columnHeight;
            nf               = runParams.numCategories;
            phase            = 6;
            writeStep        = -1;
            initialWriteTime = -1;
            InitVType        = "ZeroV";
         }
      );

end

pv.addGroup(pvClassifier, "Bias", {
         groupType        = "ConstantLayer";
         nxScale          = 1 / runParams.columnWidth;
         nyScale          = 1 / runParams.columnHeight;
         nf               = 1;
         initV            = "ConstantV";
         valueV           = runParams.biasValue;
         phase            = 0;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvClassifier, "EstimateError", {
         groupType        = "HyPerLayer";
         nxScale          = 1 / runParams.columnWidth;
         nyScale          = 1 / runParams.columnHeight;
         nf               = runParams.numCategories;
         phase            = 6;
         writeStep        = -1;
         initialWriteTime = -1;
         InitVType        = "ZeroV";
      }
   );

pv.addGroup(pvClassifier, "GroundTruth", {
         groupType        = "PvpLayer";
         nxScale          = 1 / runParams.columnWidth;
         nyScale          = 1 / runParams.columnHeight;
         nf               = runParams.numCategories;
         phase            = 0;
         displayPeriod    = 1;
         batchMethod      = "random";
         randomSeed       = 5;
         writeStep        = -1;
         initialWriteTime = -1;
         InitVType        = "ZeroV";
      }
   );

if runParams.allHiddenLayer then
   pv.addGroup(pvClassifier, "AllHiddenError", {
            groupType        = "MaskLayer";
            nxScale          = 1 / runParams.columnWidth;
            nyScale          = 1 / runParams.columnHeight;
            nf               = runParams.allHiddenFeatures;
            phase            = 7;
            writeStep        = -1;
            initialWriteTime = -1;
            maskLayerName    = "AllHidden";
            maskMethod       = "layer";
            InitVType        = "ZeroV";
            sparseLayer      = true;
         }
      );

   pv.addGroup(pvClassifier, "AllHidden", {
            groupType        = "DropoutLayer";
            nxScale          = 1 / runParams.columnWidth;
            nyScale          = 1 / runParams.columnHeight;
            nf               = runParams.allHiddenFeatures;
            phase            = 3;
            writeStep        = -1;
            initialWriteTime = -1;
            VThresh          = 0;
            AMin             = 0;
            AMax             = infinity;
            AShift           = 0;
            probability      = runParams.hiddenDropout;
            InitVType        = "ZeroV";
            sparseLayer      = true;
         }
      );
end

for index, layerName in pairs(runParams.layersToClassify) do

   if runParams.layersToClassifyXScale[layerName] == 1/runParams.columnWidth and runParams.layersToClassifyYScale[layerName] == 1/runParams.columnHeight then
      pv.addGroup(pvClassifier, layerName, {
               groupType              = "PvpLayer";
               nxScale                = runParams.layersToClassifyXScale[layerName];
               nyScale                = runParams.layersToClassifyYScale[layerName];
               nf                     = runParams.layersToClassifyFeatures[layerName];
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
               sparseLayer      = true;
            }
         );
      pv.addGroup(pvClassifier, layerName .. "MaxPool", {
               groupType          = "DropoutLayer";
               nxScale            = runParams.layersToClassifyXScale[layerName];
               nyScale            = runParams.layersToClassifyYScale[layerName];
               nf                 = runParams.layersToClassifyFeatures[layerName];
               phase              = 1;
               writeStep          = -1;
               initialWriteTime   = -1;
               VThresh            = -infinity;
               AMin               = -infinity;
               AMax               = infinity;
               AShift             = 0;
               probability        = runParams.inputDropout;
               InitVType          = "ZeroV";
               sparseLayer        = true;
            }
         );

      pv.addGroup(pvClassifier, layerName .. "HiddenError", {
               groupType        = "MaskLayer";
               nxScale          = runParams.layersToClassifyXScale[layerName];
               nyScale          = runParams.layersToClassifyYScale[layerName];
               nf               = runParams.hiddenFeatures;
               phase            = 8;
               writeStep        = -1;
               initialWriteTime = -1;
               maskLayerName    = layerName .. "Hidden";
               maskMethod       = "layer";
               InitVType        = "ZeroV";
               sparseLayer      = true;
            }
         );

      pv.addGroup(pvClassifier, layerName .. "Hidden", {
               groupType        = "DropoutLayer";
               nxScale          = runParams.layersToClassifyXScale[layerName];
               nyScale          = runParams.layersToClassifyYScale[layerName];
               nf               = runParams.hiddenFeatures;
               phase            = 2;
               writeStep        = -1;
               initialWriteTime = -1;
               VThresh          = 0;
               AMin             = 0;
               AMax             = infinity;
               AShift           = 0;
               probability      = runParams.hiddenDropout;
               InitVType        = "ZeroV";
               sparseLayer      = true;
            }
         );
   else
      pv.addGroup(pvClassifier, layerName, {
               groupType              = "PvpLayer";
               nxScale                = runParams.maxPoolX / runParams.columnWidth;
               nyScale                = runParams.maxPoolY / runParams.columnHeight;
               nf                     = runParams.layersToClassifyFeatures[layerName];
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
               sparseLayer      = true;
            }
         );
      pv.addGroup(pvClassifier, layerName .. "MaxPool", {
               groupType          = "DropoutLayer";
               nxScale            = runParams.maxPoolX / runParams.columnWidth;
               nyScale            = runParams.maxPoolY / runParams.columnHeight;
               nf                 = runParams.layersToClassifyFeatures[layerName];
               phase              = 1;
               writeStep          = -1;
               initialWriteTime   = -1;
               VThresh            = -infinity;
               AMin               = -infinity;
               AMax               = infinity;
               AShift             = 0;
               probability        = runParams.inputDropout;
               InitVType          = "ZeroV";
               sparseLayer        = true;
            }
         );

      pv.addGroup(pvClassifier, layerName .. "HiddenError", {
               groupType        = "MaskLayer";
               nxScale          = runParams.hiddenXScale * runParams.maxPoolX / runParams.columnWidth;
               nyScale          = runParams.hiddenYScale * runParams.maxPoolY / runParams.columnHeight;
               nf               = runParams.hiddenFeatures;
               phase            = 8;
               writeStep        = -1;
               initialWriteTime = -1;
               maskLayerName    = layerName .. "Hidden";
               maskMethod       = "layer";
               InitVType        = "ZeroV";
               sparseLayer      = true;
            }
         );

      pv.addGroup(pvClassifier, layerName .. "Hidden", {
               groupType        = "DropoutLayer";
               nxScale          = runParams.hiddenXScale * runParams.maxPoolX / runParams.columnWidth;
               nyScale          = runParams.hiddenYScale * runParams.maxPoolY / runParams.columnHeight;
               nf               = runParams.hiddenFeatures;
               phase            = 2;
               writeStep        = -1;
               initialWriteTime = -1;
               VThresh          = 0;
               AMin             = 0;
               AMax             = infinity;
               AShift           = 0;
               probability      = runParams.hiddenDropout;
               InitVType        = "ZeroV";
               sparseLayer      = true;
            }
         );
   end

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

if runParams.enableSimpleClassifier then
   pv.addGroup(pvClassifier, "GroundTruthToSimpleEstimateError", {
            groupType     = "IdentConn";
            channelCode   = 0;
            preLayerName  = "GroundTruth";
            postLayerName = "SimpleEstimateError";
         }
      );

   pv.addGroup(pvClassifier, "SimpleSoftmaxEstimateToSimpleEstimateError", {
            groupType     = "IdentConn";
            channelCode   = 1;
            preLayerName  = "SimpleSoftmaxEstimate";
            postLayerName = "SimpleEstimateError";
         }
      );

   pv.addGroup(pvClassifier, "BiasToSimpleEstimateError", {
            groupType               = runParams.connectionType;
            momentumMethod          = runParams.momentumType;
            momentumTau             = runParams.momentum;
            momentumDecay           = runParams.decayFactor * runParams.learningRate / 2;
            channelCode             = -1;
            preLayerName            = "Bias";
            postLayerName           = "SimpleEstimateError";
            plasticityFlag          = true;
            nxp                     = 1;
            nyp                     = 1;
            nfp                     = runParams.numCategories;
            dWMax                   = runParams.learningRate / 2;
            weightInitType          = "GaussianRandomWeight";
            wGaussMean              = 0;
            wGaussStdev             = 0;
            normalizeMethod         = runParams.normType;
            strength                = runParams.normStrength;
            normalizeDw             = runParams.normDW;
            weightUpdatePeriod      = 1;
            dWMaxDecayFactor        = runParams.learningRateDecay; 
         }
      );

   pv.addGroup(pvClassifier, "BiasToSimpleCategoryEstimate", {
            groupType        = "CloneConn";
            channelCode      = 0;
            preLayerName     = "Bias";
            postLayerName    = "SimpleCategoryEstimate";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = "BiasToSimpleEstimateError";
         }
      );
end

if runParams.allHiddenLayer then
   pv.addGroup(pvClassifier, "AllHiddenToEstimateError", {
            groupType               = runParams.connectionType;
            momentumMethod          = runParams.momentumType;
            momentumTau             = runParams.momentum;
            momentumDecay           = runParams.decayFactor * runParams.learningRate;
            channelCode             = -1;
            preLayerName            = "AllHidden";
            postLayerName           = "EstimateError";
            plasticityFlag          = true;
            nxp                     = 1;
            nyp                     = 1;
            nfp                     = runParams.numCategories;
            dWMax                   = runParams.learningRate;
            weightInitType          = "UniformRandomWeight";
            wMinInit                = -runParams.weightStd;
            wMaxInit                = runParams.weightStd;
            sparseFraction          = 0.9;
            minNNZ                  = 1; 
            wGaussMean              = 0;
            wGaussStdev             = runParams.weightStd;
            normalizeMethod         = runParams.normType;
            strength                = runParams.normStrength;
            normalizeDw             = runParams.normDW;
            receiveGpu              = runParams.useGpu;
            initialWeightUpdateTime = 2;
            writeStep               = runParams.debugWriteStep;
            initialWriteTime        = 0;
            weightUpdatePeriod      = 1;
            dWMaxDecayFactor        = runParams.learningRateDecay; 
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
            groupType               = runParams.connectionType;
            momentumMethod          = runParams.momentumType;
            momentumTau             = runParams.momentum;
            momentumDecay           = runParams.decayFactor * runParams.learningRate / 2;
            channelCode             = -1;
            preLayerName            = "Bias";
            postLayerName           = "AllHiddenError";
            plasticityFlag          = true;
            nxp                     = 1;
            nyp                     = 1;
            nfp                     = runParams.allHiddenFeatures;
            dWMax                   = runParams.learningRate / 2;
            weightInitType          = "GaussianRandomWeight";
            wGaussMean              = 0;
            wGaussStdev             = 0;
            normalizeMethod         = runParams.normType;
            strength                = runParams.normStrength;
            normalizeDw             = runParams.normDW;
            weightUpdatePeriod      = 1;
            dWMaxDecayFactor        = runParams.learningRateDecay; 
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
         groupType               = runParams.connectionType;
         momentumMethod          = runParams.momentumType;
         momentumTau             = runParams.momentum;
         momentumDecay           = runParams.decayFactor * runParams.learningRate / 2;
         channelCode             = -1;
         preLayerName            = "Bias";
         postLayerName           = "EstimateError";
         plasticityFlag          = true;
         nxp                     = 1;
         nyp                     = 1;
         nfp                     = runParams.numCategories;
         dWMax                   = runParams.learningRate / 2;
         weightInitType          = "GaussianRandomWeight";
         wGaussMean              = 0;
         wGaussStdev             = 0;
         normalizeMethod         = runParams.normType;
         strength                = runParams.normStrength;
         normalizeDw             = runParams.normDW;
         weightUpdatePeriod      = 1;
         dWMaxDecayFactor        = runParams.learningRateDecay; 
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


for index, layerName in pairs(runParams.layersToClassify) do
   local maxPoolLayerName = layerName .. "MaxPool";

   if runParams.allHiddenLayer then
      pv.addGroup(pvClassifier, layerName .. "HiddenToAllHiddenError", {
               groupType               = runParams.connectionType;
               momentumMethod          = runParams.momentumType;
               momentumTau             = runParams.momentum;
               momentumDecay           = runParams.decayFactor * runParams.learningRate;
               channelCode             = -1;
               preLayerName            = layerName .. "Hidden";
               postLayerName           = "AllHiddenError";
               plasticityFlag          = true;
               nxp                     = 1;
               nyp                     = 1;
               nfp                     = runParams.allHiddenFeatures;
               dWMax                   = runParams.learningRate;
               weightInitType          = "UniformRandomWeight";
               wMinInit                = -runParams.weightStd;
               wMaxInit                = runParams.weightStd;
               sparseFraction          = 0.9;
               minNNZ                  = 1; 
               normalizeMethod         = runParams.normType;
               strength                = runParams.normStrength;
               normalizeDw             = runParams.normDW;
               receiveGpu              = runParams.useGpu;
               weightUpdatePeriod      = 1;
               dWMaxDecayFactor        = runParams.learningRateDecay; 
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

      pv.addGroup(pvClassifier, maxPoolLayerName .. "ToAllHiddenError", {
            groupType               = runParams.connectionType;
            momentumMethod          = runParams.momentumType;
            momentumTau             = runParams.momentum;
            momentumDecay           = runParams.decayFactor * runParams.rateFactor * runParams.learningRate;
            channelCode             = -1;
            preLayerName            = maxPoolLayerName;
            postLayerName           = "AllHiddenError";
            plasticityFlag          = true;
            nxp                     = 1;
            nyp                     = 1;
            dWMax                   = runParams.rateFactor * runParams.learningRate;
            weightInitType          = "UniformRandomWeight";
            wMinInit                = -runParams.weightStd;
            wMaxInit                = runParams.weightStd;
            sparseFraction          = 0.9;
            minNNZ                  = 1; 
            receiveGpu              = runParams.useGpu;
            normalizeMethod         = runParams.normType;
            strength                = runParams.normStrength;
            normalizeDw             = runParams.normDW;
            sharedWeights           = runParams.sharedWeights;
            weightUpdatePeriod      = 1;
            dWMaxDecayFactor        = runParams.learningRateDecay; 
         }
      );


      pv.addGroup(pvClassifier, maxPoolLayerName .. "ToAllHidden", {
            groupType        = "CloneConn";
            channelCode      = 0;
            preLayerName     = maxPoolLayerName;
            postLayerName    = "AllHidden";
            writeStep        = -1;
            initialWriteTime = -1;
            originalConnName = maxPoolLayerName .. "ToAllHiddenError";
         }
      );

   end
   pv.addGroup(pvClassifier, layerName .. "HiddenToEstimateError", {
            groupType               = runParams.connectionType;
            momentumMethod          = runParams.momentumType;
            momentumTau             = runParams.momentum;
            momentumDecay           = runParams.decayFactor * runParams.learningRate;
            channelCode             = -1;
            preLayerName            = layerName .. "Hidden";
            postLayerName           = "EstimateError";
            plasticityFlag          = true;
            nxp                     = 1;
            nyp                     = 1;
            nfp                     = runParams.numCategories;
            dWMax                   = runParams.learningRate;
            weightInitType          = "UniformRandomWeight";
            wMinInit                = -runParams.weightStd;
            wMaxInit                = runParams.weightStd;
            sparseFraction          = 0.9;
            minNNZ                  = 1; 
            normalizeMethod         = runParams.normType;
            strength                = runParams.normStrength;
            normalizeDw             = runParams.normDW;
            receiveGpu              = runParams.useGpu;
            weightUpdatePeriod      = 1;
            dWMaxDecayFactor        = runParams.learningRateDecay; 
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


   if runParams.enableSimpleClassifier then
      pv.addGroup(pvClassifier, maxPoolLayerName .. "ToSimpleEstimateError", {
               groupType               = runParams.connectionType;
               momentumMethod          = runParams.momentumType;
               momentumTau             = runParams.momentum;
               momentumDecay           = runParams.decayFactor * runParams.learningRate;
               channelCode             = -1;
               preLayerName            = maxPoolLayerName;
               postLayerName           = "SimpleEstimateError";
               plasticityFlag          = true;
               nxp                     = 1;
               nyp                     = 1;
               nfp                     = runParams.numCategories;
               dWMax                   = runParams.learningRate;
               weightInitType          = "UniformRandomWeight";
               wMinInit                = -runParams.weightStd;
               wMaxInit                = runParams.weightStd;
               sparseFraction          = 0.9;
               minNNZ                  = 1; 
               receiveGpu              = runParams.useGpu;
               normalizeMethod         = runParams.normType;
               strength                = runParams.normStrength;
               normalizeDw             = runParams.normDW;
               sharedWeights           = true;
               weightUpdatePeriod      = 1;
               dWMaxDecayFactor        = runParams.learningRateDecay; 
            }
         );

      pv.addGroup(pvClassifier, maxPoolLayerName .. "ToSimpleCategoryEstimate", {
               groupType        = "CloneConn";
               channelCode      = 0;
               preLayerName     = maxPoolLayerName;
               postLayerName    = "SimpleCategoryEstimate";
               writeStep        = -1;
               initialWriteTime = -1;
               originalConnName = maxPoolLayerName .. "ToSimpleEstimateError";
            }
         );
   end
   
   pv.addGroup(pvClassifier, layerName .. "To" .. maxPoolLayerName, {
            groupType             = "IdentConn";
            channelCode           = 0;
            preLayerName          = layerName;
            postLayerName         = maxPoolLayerName;
            writeStep             = -1;
         }
      );

   pv.addGroup(pvClassifier, maxPoolLayerName .. "To" .. layerName .. "HiddenError", {
            groupType               = runParams.connectionType;
            momentumMethod          = runParams.momentumType;
            momentumTau             = runParams.momentum;
            momentumDecay           = runParams.decayFactor * runParams.rateFactor * runParams.learningRate;
            channelCode             = -1;
            preLayerName            = maxPoolLayerName;
            postLayerName           = layerName .. "HiddenError";
            plasticityFlag          = true;
            nxp                     = math.min(runParams.layersToClassifyXScale[layerName] * runParams.columnWidth, runParams.hiddenPatch);
            nyp                     = math.min(runParams.layersToClassifyYScale[layerName] * runParams.columnHeight, runParams.hiddenPatch);
            nfp                     = runParams.hiddenFeatures;
            dWMax                   = runParams.rateFactor * runParams.learningRate;
            weightInitType          = "UniformRandomWeight";
            wMinInit                = -runParams.weightStd;
            wMaxInit                = runParams.weightStd;
            sparseFraction          = 0.9;
            minNNZ                  = 1; 
            receiveGpu              = runParams.useGpu;
            normalizeMethod         = runParams.normType;
            strength                = runParams.normStrength;
            normalizeDw             = runParams.normDW;
            sharedWeights           = runParams.sharedWeights;
            weightUpdatePeriod      = 1;
            dWMaxDecayFactor        = runParams.learningRateDecay; 
         }
      );

-- Pretty sure this is supported now, uncomment if not
--   if not runParams.sharedWeights then
--      pvClassifier[maxPoolLayerName .. "To" .. layerName .. "HiddenError"].receiveGpu = false;
--   end

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
            groupType               = runParams.connectionType;
            momentumMethod          = runParams.momentumType;
            momentumTau             = runParams.momentum;
            momentumDecay           = runParams.decayFactor * runParams.rateFactor * runParams.learningRate / 2;
            channelCode             = -1;
            preLayerName            = "Bias";
            postLayerName           = layerName .. "HiddenError";
            plasticityFlag          = true;
            nxp                     = math.min(runParams.hiddenXScale * runParams.maxPoolX, runParams.layersToClassifyXScale[layerName] * runParams.columnWidth);
            nyp                     = math.min(runParams.hiddenYScale * runParams.maxPoolY, runParams.layersToClassifyYScale[layerName] * runParams.columnHeight);
            nfp                     = runParams.hiddenFeatures;
            dWMax                   = runParams.rateFactor * runParams.learningRate / 2;
            weightInitType          = "GaussianRandomWeight";
            wGaussMean              = 0;
            wGaussStdev             = 0;
            normalizeMethod         = runParams.normType;
            strength                = runParams.normStrength;
            normalizeDw             = runParams.normDW;
            sharedWeights           = runParams.sharedWeights;
            weightUpdatePeriod      = 1;
            dWMaxDecayFactor        = runParams.learningRateDecay; 
         }
      );

--   if not runParams.sharedWeights then
--      pvClassifier["BiasTo" .. layerName .. "HiddenError"].receiveGpu = false;
--   end


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
