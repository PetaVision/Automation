------------
-- Params --
------------


local useGpu           = true;
local inputFeatures    = 3;
local nbatch           = numSparseBatches;
local plasticityFlag   = true;
local checkpointPeriod = displayPeriod * 25;
local patchSize        = 7;
local stride           = 1;

local dictionarySize   = 64;
if globalDictionarySize ~= nil then
   dictionarySize = globalDictionarySize;
end

local VThresh          = 0.15;
if globalVThresh ~= nil then
   VThresh = globalVThresh;
end

local dWMax            = 0.01;
local momentumTau      = 500;
local AMin             = 0;
local AMax             = infinity;
local AShift           = 0;
local VWidth           = 0; 
local timeConstantTau  = 125;
local weightInit       = 1.0;
local sparseFraction   = 0.975;


-- This file requires the global variables displayPeriod,
-- columnWidth, and columnHeight to already be set.

------------
-- Column --
------------

local pvParams = {
  column = {
      groupType                     = "HyPerCol";
      startTime                     = 0;
      dt                            = 1;
      progressInterval              = displayPeriod;
      randomSeed                    = 1234567890;
      nx                            = columnWidth;
      ny                            = columnHeight;
      nbatch                        = nbatch;
      checkpointWrite               = true;
      checkpointWriteTriggerMode    = "step";
      checkpointWriteStepInterval   = checkpointPeriod;
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
         nf                     = inputFeatures;
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
         nf               = inputFeatures;
         phase            = 2;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvParams, "S1", {
         groupType              = "HyPerLCALayer";
         nxScale                = 1/stride;
         nyScale                = 1/stride;
         nf                     = dictionarySize;
         phase                  = 3;
         InitVType              = "ConstantV";
         valueV                 = VThresh / 2;
         triggerLayerName       = NULL;
         sparseLayer            = true;
         writeSparseValues      = true;
         updateGpu              = useGpu;
         dataType               = nil;
         VThresh                = VThresh;
         AMin                   = AMin;
         AMax                   = AMax;
         AShift                 = AShift;
         VWidth                 = VWidth;
         timeConstantTau        = timeConstantTau;
         selfInteract           = true;
         adaptiveTimeScaleProbe = "AdaptProbe";
      }
   );

pv.addGroup(pvParams, "ImageReconS1",  {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = inputFeatures;
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
         scale         = weightInit;
      }
   );

pv.addGroup(pvParams, "ImageReconS1ErrorToS1", {
         groupType                     = "TransposeConn";
         preLayerName                  = "ImageReconS1Error";
         postLayerName                 = "S1";
         channelCode                   = 0;
         receiveGpu                    = useGpu;
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
         plasticityFlag          = plasticityFlag;
         sharedWeights           = true;
         weightInitType          = "UniformRandomWeight";
         wMinInit                = -1;
         wMaxInit                = 1;
         minNNZ                  = 1;
         sparseFraction          = sparseFraction;
         triggerLayerName        = "Image";
         pvpatchAccumulateType   = "convolve";
         nxp                     = patchSize;
         nyp                     = patchSize;
         normalizeMethod         = "normalizeL2";
         strength                = 1;
         normalizeOnInitialize   = true;
         normalizeOnWeightUpdate = true;
         minL2NormTolerated      = 0;
         dWMax                   = dWMax; 
         momentumTau             = momentumTau;
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
         coefficient     = VThresh;
         maskLayerName   = nil;
      }
   );

-- Return our table. The file that calls this
-- one does the actual writing to disk.
return pvParams;
