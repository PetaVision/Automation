------------
-- Column --
------------

local useProbes = true;

if runParams.xstride == nil or runParams.ystride == nil then
   runParams.xstride = runParams.stride;
   runParams.ystride = runParams.stride;
end

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
      deleteOlderCheckpoints        = false;
   } 
};

------------
-- Layers --
------------

pv.addGroup(pvParams, "Image",  {
         groupType              = "PvpLayer";
         nxScale                = 1;
         nyScale                = 1;
         nf                     = runParams.inputFeatures;
         phase                  = 1;
         writeStep              = -1;
         initialWriteTime       = -1;
         offsetAnchor           = "tl";
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
         nxScale                = 1/runParams.xstride;
         nyScale                = 1/runParams.ystride;
         nf                     = runParams.dictionarySize;
         phase                  = 3;
         InitVType              = "ConstantV";
         valueV                 = runParams.VThresh;
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
         adaptiveTimeScaleProbe = "AdaptProbe";
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
         nxp                     = math.min(runParams.columnWidth, runParams.patchSize);
         nyp                     = math.min(runParams.columnHeight, runParams.patchSize);
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
            growthFactor     = 0.03;
            writeTimeScales  = true;
            kneeThresh       = 0.225;
            kneeSlope        = 0.015;
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
