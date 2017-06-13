------------

-- S2 Params
local strideS2   = 2;
local strideS3   = 16;
local patchS2    = runParams.patchSize + 1;
local patchS3    = 16; 
local imgPatchS2 = runParams.patchSize + patchS2 - 1;
local imgPatchS3 = 32;
local dictionaryS3 = 1024;

------------
-- Column --
------------

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
      deleteOlderCheckpoints        = true;
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

pv.addGroup(pvParams, "ImageReconError", {
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
         adaptiveTimeScaleProbe = "AdaptProbe";
      }
   );

pv.addGroup(pvParams, "S1V", {
         groupType              = "CloneVLayer";
         nxScale                = 1/runParams.stride;
         nyScale                = 1/runParams.stride;
         nf                     = runParams.dictionarySize;
         phase                  = 4;
         InitVType              = "ConstantV";
         valueV                 = runParams.VThresh / 2;
         triggerLayerName       = NULL;
         originalLayerName      = "S1";
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

pv.addGroup(pvParams, "S1VReconError", {
         groupType        = "HyPerLayer";
         nxScale          = 1/runParams.stride;
         nyScale          = 1/runParams.stride;
         nf               = runParams.dictionarySize;
         phase            = 9;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );


pv.addGroup(pvParams, "S2", {
         groupType              = "HyPerLCALayer";
         nxScale                = 1/runParams.stride * 1/strideS2;
         nyScale                = 1/runParams.stride * 1/strideS2;
         nf                     = runParams.dictionarySize * strideS2 * strideS2;
         phase                  = 5;
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
         adaptiveTimeScaleProbe = "AdaptProbe";
      }
   );

pv.addGroup(pvParams, "S2V", {
         groupType              = "CloneVLayer";
         nxScale                = 1/runParams.stride * 1/strideS2;
         nyScale                = 1/runParams.stride * 1/strideS2;
         nf                     = runParams.dictionarySize * strideS2 * strideS2;
         phase                  = 6;
         InitVType              = "ConstantV";
         valueV                 = runParams.VThresh / 2;
         triggerLayerName       = NULL;
         originalLayerName      = "S2";
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

pv.addGroup(pvParams, "S2VReconError", {
         groupType        = "HyPerLayer";
         nxScale          = 1/runParams.stride * 1/strideS2;
         nyScale          = 1/runParams.stride * 1/strideS2;
         nf               = runParams.dictionarySize * strideS2 * strideS2;
         phase            = 10;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

 pv.addGroup(pvParams, "S3", {
          groupType              = "HyPerLCALayer";
          nxScale                = 1/runParams.stride * 1/strideS2 * 1/strideS3;
          nyScale                = 1/runParams.stride * 1/strideS2 * 1/strideS3;
          nf                     = dictionaryS3; --runParams.dictionarySize * strideS2 * strideS2 * strideS3 * strideS3;
          phase                  = 7;
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

pv.addGroup(pvParams, "ImageReconS2",  {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = runParams.inputFeatures;
         phase            = 6;
         InitVType        = "ZeroV";
         triggerLayerName = NULL;
         writeStep        = -1;
         initialWriteTime = -1;
         sparseLayer      = false;
         updateGpu        = false;
         dataType         = nil;
      }
   );

pv.addGroup(pvParams, "ImageReconS3",  {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = runParams.inputFeatures;
         phase            = 8;
         InitVType        = "ZeroV";
         triggerLayerName = NULL;
         writeStep        = -1;
         initialWriteTime = -1;
         sparseLayer      = false;
         updateGpu        = false;
         dataType         = nil;
      }
   );


pv.addGroup(pvParams, "S1VReconS2",  {
         groupType        = "HyPerLayer";
         nxScale          = 1/runParams.stride;
         nyScale          = 1/runParams.stride;
         nf               = runParams.dictionarySize;
         phase            = 6;
         InitVType        = "ZeroV";
         triggerLayerName = NULL;
         writeStep        = -1;
         initialWriteTime = -1;
         sparseLayer      = false;
         updateGpu        = false;
         dataType         = nil;
      }
   );

pv.addGroup(pvParams, "S2VReconS3",  {
         groupType        = "HyPerLayer";
         nxScale          = 1/runParams.stride * 1/strideS2;
         nyScale          = 1/runParams.stride * 1/strideS2;
         nf               = runParams.dictionarySize * strideS2 * strideS2;
         phase            = 8;
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

pv.addGroup(pvParams, "ImageToImageReconError", {
         groupType     = "RescaleConn";
         preLayerName  = "Image";
         postLayerName = "ImageReconError";
         channelCode   = 0;
         scale         = runParams.weightInit;
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

pv.addGroup(pvParams, "ImageReconErrorToS1", {
         groupType                     = "TransposeConn";
         preLayerName                  = "ImageReconError";
         postLayerName                 = "S1";
         channelCode                   = 0;
         receiveGpu                    = runParams.useGpu;
         updateGSynFromPostPerspective = true;
         pvpatchAccumulateType         = "convolve";
         writeStep                     = -1;
         originalConnName              = "S1ToImageReconError";
      }
   );

pv.addGroup(pvParams, "ImageReconErrorToS2", {
         groupType                     = "TransposeConn";
         preLayerName                  = "ImageReconError";
         postLayerName                 = "S2";
         channelCode                   = 0;
         receiveGpu                    = runParams.useGpu;
         updateGSynFromPostPerspective = true;
         pvpatchAccumulateType         = "convolve";
         writeStep                     = -1;
         originalConnName              = "S2ToImageReconError";
      }
   );

pv.addGroup(pvParams, "ImageReconErrorToS3", {
         groupType                     = "TransposeConn";
         preLayerName                  = "ImageReconError";
         postLayerName                 = "S3";
         channelCode                   = 0;
         receiveGpu                    = runParams.useGpu;
         updateGSynFromPostPerspective = true;
         pvpatchAccumulateType         = "convolve";
         writeStep                     = -1;
         originalConnName              = "S3ToImageReconError";
      }
   );

pv.addGroup(pvParams, "S1ToImageReconError", {
         groupType               = "MomentumConn";
         preLayerName            = "S1";
         postLayerName           = "ImageReconError";
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

pv.addGroup(pvParams, "S2ToImageReconError", {
         groupType               = "MomentumConn";
         preLayerName            = "S2";
         postLayerName           = "ImageReconError";
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
         nxp                     = imgPatchS2;
         nyp                     = imgPatchS2;
         normalizeMethod         = "normalizeGroup";
	 normalizeGroupName      = "S2ToS1VReconError";
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

pv.addGroup(pvParams, "S3ToImageReconError", {
         groupType               = "MomentumConn";
         preLayerName            = "S3";
         postLayerName           = "ImageReconError";
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
         nxp                     = imgPatchS3;
         nyp                     = imgPatchS3;
         normalizeMethod         = "normalizeGroup";
	 normalizeGroupName      = "S3ToS2VReconError";
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

pv.addGroup(pvParams, "S2ToS1VReconError", {
         groupType               = "MomentumConn";
         preLayerName            = "S2";
         postLayerName           = "S1VReconError";
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
         nxp                     = patchS2;
         nyp                     = patchS2;
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

pv.addGroup(pvParams, "S1VReconErrorToS2", {
         groupType                     = "TransposeConn";
         preLayerName                  = "S1VReconError";
         postLayerName                 = "S2";
         channelCode                   = 0;
         receiveGpu                    = runParams.useGpu;
         updateGSynFromPostPerspective = true;
         pvpatchAccumulateType         = "convolve";
         writeStep                     = -1;
         originalConnName              = "S2ToS1VReconError";
      }
   );

pv.addGroup(pvParams, "S3ToS2VReconError", {
         groupType               = "MomentumConn";
         preLayerName            = "S3";
         postLayerName           = "S2VReconError";
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
         nxp                     = patchS3;
         nyp                     = patchS3;
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

pv.addGroup(pvParams, "S2VReconErrorToS3", {
         groupType                     = "TransposeConn";
         preLayerName                  = "S2VReconError";
         postLayerName                 = "S3";
         channelCode                   = 0;
         receiveGpu                    = runParams.useGpu;
         updateGSynFromPostPerspective = true;
         pvpatchAccumulateType         = "convolve";
         writeStep                     = -1;
         originalConnName              = "S3ToS2VReconError";
      }
   );


pv.addGroup(pvParams, "S1ToImageReconS1", {
         groupType             = "CloneConn";
         preLayerName          = "S1";
         postLayerName         = "ImageReconS1";
         channelCode           = 0;
         pvpatchAccumulateType = "convolve";
         originalConnName      = "S1ToImageReconError";
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

pv.addGroup(pvParams, "S2ToImageReconS2", {
         groupType             = "CloneConn";
         preLayerName          = "S2";
         postLayerName         = "ImageReconS2";
         channelCode           = 0;
         pvpatchAccumulateType = "convolve";
         originalConnName      = "S2ToImageReconError";
      }
   );

pv.addGroup(pvParams, "S2ToS1VReconS2", {
         groupType             = "CloneConn";
         preLayerName          = "S2";
         postLayerName         = "S1VReconS2";
         channelCode           = 0;
         pvpatchAccumulateType = "convolve";
         originalConnName      = "S2ToS1VReconError";
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

pv.addGroup(pvParams, "S3ToImageReconS3", {
         groupType             = "CloneConn";
         preLayerName          = "S3";
         postLayerName         = "ImageReconS3";
         channelCode           = 0;
         pvpatchAccumulateType = "convolve";
         originalConnName      = "S3ToImageReconError";
      }
   );

pv.addGroup(pvParams, "S3ToS2VReconS3", {
         groupType             = "CloneConn";
         preLayerName          = "S3";
         postLayerName         = "S2VReconS3";
         channelCode           = 0;
         pvpatchAccumulateType = "convolve";
         originalConnName      = "S3ToS2VReconError";
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );


pv.addGroup(pvParams, "ImageReconS1ToImageReconError", {
         groupType     = "IdentConn";
         preLayerName  = "ImageReconS1";
         postLayerName = "ImageReconError";
         channelCode   = 1;
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

pv.addGroup(pvParams, "ImageReconS2ToImageReconError", {
         groupType     = "IdentConn";
         preLayerName  = "ImageReconS2";
         postLayerName = "ImageReconError";
         channelCode   = 1;
      }
   );

pv.addGroup(pvParams, "ImageReconS3ToImageReconError", {
         groupType     = "IdentConn";
         preLayerName  = "ImageReconS3";
         postLayerName = "ImageReconError";
         channelCode   = 1;
      }
   );

pv.addGroup(pvParams, "S1VReconS2ToS1VReconError", {
         groupType     = "IdentConn";
         preLayerName  = "S1VReconS2";
         postLayerName = "S1VReconError";
         channelCode   = 1;
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

pv.addGroup(pvParams, "S1VToS1VReconError", {
         groupType     = "IdentConn";
         preLayerName  = "S1V";
         postLayerName = "S1VReconError";
         channelCode   = 0;
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

pv.addGroup(pvParams, "S1VReconErrorToS1", {
         groupType     = "IdentConn";
         preLayerName  = "S1VReconError";
         postLayerName = "S1";
         channelCode   = 1;
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

pv.addGroup(pvParams, "S2VReconS3ToS2VReconError", {
         groupType     = "IdentConn";
         preLayerName  = "S2VReconS3";
         postLayerName = "S2VReconError";
         channelCode   = 1;
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

pv.addGroup(pvParams, "S2VToS2VReconError", {
         groupType     = "IdentConn";
         preLayerName  = "S2V";
         postLayerName = "S2VReconError";
         channelCode   = 0;
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

pv.addGroup(pvParams, "S2VReconErrorToS2", {
         groupType     = "IdentConn";
         preLayerName  = "S2VReconError";
         postLayerName = "S2";
         channelCode   = 1;
         initialWriteTime       = -1;
         writeStep              = -1;
      }
   );

------------
-- Probes --
------------

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
         tauFactor        = 0.025;
         growthFactor     = 0.02;
         writeTimeScales  = true;
         kneeThresh       = 0.15;
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

pv.addGroup(pvParams, "ImageReconErrorL2NormEnergyProbe", {
         groupType       = "L2NormProbe";
         targetLayer     = "ImageReconError";
         message         = nil;
         textOutputFlag  = true;
         probeOutputFile = "ImageReconErrorL2.txt";
         energyProbe     = "EnergyProbe";
         coefficient     = 0.5;
         maskLayerName   = nil;
         exponent        = 2;
      }
   );

pv.addGroup(pvParams, "S1ReconErrorL2NormEnergyProbe", {
         groupType       = "L2NormProbe";
         targetLayer     = "S1VReconError";
         message         = nil;
         textOutputFlag  = true;
         probeOutputFile = "S1ReconErrorL2.txt";
         energyProbe     = "EnergyProbe";
         coefficient     = 0.5;
         maskLayerName   = nil;
         exponent        = 2;
      }
   );

pv.addGroup(pvParams, "S2ReconErrorL2NormEnergyProbe", {
         groupType       = "L2NormProbe";
         targetLayer     = "S2VReconError";
         message         = nil;
         textOutputFlag  = true;
         probeOutputFile = "S2ReconErrorL2.txt";
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

pv.addGroup(pvParams, "S2L1NormEnergyProbe", {
         groupType       = "L1NormProbe";
         targetLayer     = "S2";
         message         = nil;
         textOutputFlag  = true;
         probeOutputFile = "S2L1.txt";
         energyProbe     = "EnergyProbe";
         coefficient     = runParams.VThresh;
         maskLayerName   = nil;
      }
   );

pv.addGroup(pvParams, "S3L1NormEnergyProbe", {
         groupType       = "L1NormProbe";
         targetLayer     = "S3";
         message         = nil;
         textOutputFlag  = true;
         probeOutputFile = "S3L1.txt";
         energyProbe     = "EnergyProbe";
         coefficient     = runParams.VThresh;
         maskLayerName   = nil;
      }
   );


-- Return our table. The file that calls this
-- one does the actual writing to disk.
return pvParams;
