------------
-- Column --
------------

local useProbes = true;

local pvParams = {
  column = {
      groupType                     = "HyPerCol";
      startTime                     = 0;
      dt                            = 1;
      progressInterval              = runParams.displayPeriod;
      randomSeed                    = 1234567890;
      nx                            = runParams.columnWidth;
      ny                            = runParams.columnHeight;
      nbatch                        = runConfig.numSparseBatches;
      checkpointWrite               = true;
      checkpointWriteTriggerMode    = "step";
      checkpointWriteStepInterval   = runParams.checkpointPeriod;
      deleteOlderCheckpoints        = false; --true;
   } 
};

------------
-- Layers --
------------

pv.addGroup(pvParams, "Image",  {
         groupType              = "ImageLayer";
         nxScale                = 1;
         nyScale                = 1;
         nf                     = runParams.inputFeatures;
         phase                  = 1;
         writeStep              = -1;
         initialWriteTime       = -1;
         offsetAnchor           = "cc";
         inverseFlag            = false;
         normalizeLuminanceFlag = true;
         normalizeStdDev        = true;
         autoResizeFlag         = false;
         batchMethod            = "byFile";
         writeFrameToTimestamp  = true;
         resetToStartOnLoop     = false;
      }
   );

pv.addGroup(pvParams, "ImageReconS1Error", {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = runParams.inputFeatures;
         phase            = 2;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvParams, "S1", {
         groupType              = "HyPerLCALayer";
         nxScale                = 1/runParams.stride;
         nyScale                = 1/runParams.stride;
         nf                     = runParams.dictionarySize;
         phase                  = 3;
         InitVType              = "ConstantV";
         valueV                 = runParams.VThresh / 2;
         triggerLayerName       = NULL;
         sparseLayer            = true;
         writeSparseValues      = true;
         updateGpu              = runParams.useGpu;
         dataType               = nil;
         VThresh                = runParams.VThresh;
         AMin                   = runParams.AMin;
         AMax                   = runParams.AMax;
         AShift                 = runParams.AShift;
         VWidth                 = runParams.VWidth;
         timeConstantTau        = runParams.timeConstantTau;
         selfInteract           = true;
         adaptiveTimeScaleProbe = nil;-- "AdaptProbe";
      }
   );

pv.addGroup(pvParams, "ImageReconS1",  {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = runParams.inputFeatures;
         phase            = 4;
         InitVType        = "ZeroV";
         triggerLayerName = NULL;
         writeStep        = -1;
         initialWriteTime = -1;
         sparseLayer      = false;
         updateGpu        = false;
         dataType         = nil;
      }
   );

-----------------
-- Connections --
-----------------

pv.addGroup(pvParams, "ImageToImageReconS1Error", {
         groupType     = "RescaleConn";
         preLayerName  = "Image";
         postLayerName = "ImageReconS1Error";
         channelCode   = 0;
         scale         = runParams.weightInit;
      }
   );

pv.addGroup(pvParams, "ImageReconS1ErrorToS1", {
         groupType                     = "TransposeConn";
         preLayerName                  = "ImageReconS1Error";
         postLayerName                 = "S1";
         channelCode                   = 0;
         receiveGpu                    = runParams.useGpu;
         updateGSynFromPostPerspective = true;
         pvpatchAccumulateType         = "convolve";
         writeStep                     = -1;
         originalConnName              = "S1ToImageReconS1Error";
      }
   );

pv.addGroup(pvParams, "S1ToImageReconS1Error", {
         groupType               = "MomentumConn";
         preLayerName            = "S1";
         postLayerName           = "ImageReconS1Error";
         channelCode             = -1;
         plasticityFlag          = runParams.plasticityFlag;
         sharedWeights           = true;
         weightInitType          = "UniformRandomWeight";
         wMinInit                = -1;
         wMaxInit                = 1;
         minNNZ                  = 1;
         sparseFraction          = runParams.sparseFraction;
         triggerLayerName        = "Image";
         pvpatchAccumulateType   = "convolve";
         nxp                     = runParams.patchSize;
         nyp                     = runParams.patchSize;
         normalizeMethod         = "normalizeL2";
         strength                = 1;
         normalizeOnInitialize   = true;
         normalizeOnWeightUpdate = true;
         minL2NormTolerated      = 0;
         dWMax                   = runParams.dWMax; 
         momentumTau             = runParams.momentumTau;
         momentumMethod          = "viscosity";
         momentumDecay           = 0;
         initialWriteTime        = -1;
         writeStep               = -1;
      }
   );

pv.addGroup(pvParams, "S1ToImageReconS1", {
         groupType             = "CloneConn";
         preLayerName          = "S1";
         postLayerName         = "ImageReconS1";
         channelCode           = 0;
         pvpatchAccumulateType = "convolve";
         originalConnName      = "S1ToImageReconS1Error";
      }
   );

pv.addGroup(pvParams, "ImageReconS1ToImageReconS1Error", {
         groupType     = "IdentConn";
         preLayerName  = "ImageReconS1";
         postLayerName = "ImageReconS1Error";
         channelCode   = 1;
      }
   );

------------
-- Probes --
------------

if useProbes then
   pv.addGroup(pvParams, "AdaptProbe", {
            groupType        = "KneeTimeScaleProbe";
            targetName       = "EnergyProbe";
            message          = NULL;
            textOutputFlag   = true;
            probeOutputFile  = "AdaptiveTimeScales.txt";
            triggerLayerName = "Image";
            triggerOffset    = 0;
            baseMax          = 0.011;
            baseMin          = 0.01;
            tauFactor        = 0.05;
            growthFactor     = 0.025;
            writeTimeScales  = true;
            kneeThresh       = 0.2;
            kneeSlope        = 0.01;
         }
      );

   pv.addGroup(pvParams, "EnergyProbe", {
            groupType        = "ColumnEnergyProbe";
            message          = nil;
            textOutputFlag   = true;
            probeOutputFile  = "EnergyProbe.txt";
            triggerLayerName = nil;
            energyProbe      = nil;
         }
      );

   pv.addGroup(pvParams, "ImageReconS1ErrorL2NormEnergyProbe", {
            groupType       = "L2NormProbe";
            targetLayer     = "ImageReconS1Error";
            message         = nil;
            textOutputFlag  = true;
            probeOutputFile = "ImageReconS1ErrorL2.txt";
            energyProbe     = "EnergyProbe";
            coefficient     = 0.5;
            maskLayerName   = nil;
            exponent        = 2;
         }
      );

   pv.addGroup(pvParams, "S1L1NormEnergyProbe", {
            groupType       = "L1NormProbe";
            targetLayer     = "S1";
            message         = nil;
            textOutputFlag  = true;
            probeOutputFile = "S1L1.txt";
            energyProbe     = "EnergyProbe";
            coefficient     = runParams.VThresh;
            maskLayerName   = nil;
         }
      );
end
-- Return our table. The file that calls this
-- one does the actual writing to disk.
return pvParams;
