------------
-- Params --
------------

local useGpu           = true;
local inputWidth       = 32;
local inputHeight      = 32;
local inputFeatures    = 3;
local tiers            = 4;
local patchSize        = {5,    6,    6,    8};  
local patchSizeATA     = {5,    10,   20,   32}; -- effective size on retina  
local stride           = {1,    2,    4,    32};
local dictionarySize   = {16,   32,   64,   512};
local weightInitRange  = {1.0,  1.0,  1.0,  1.0};
local weightSparsity   = {0.9,  0.9,  0.9,  0.9};
local plasticity       = {true, true, true, true};
local learningRate     = {1.0,  1.0,  1.0,  1.0};
local imageRecon       = true;
local topDownFeedback  = false;
local nbatch           = 4;
local checkpointPeriod = (displayPeriod * 10);
local plasticityFlag   = true;
local momentumTau      = 200;
local VThreshFactor    = 1.0;
local VThresh          = 0.1;
local AMin             = 0;
local AMax             = infinity;
local timeConstantTau  = { displayPeriod / 5,
                           displayPeriod / 5,
                           displayPeriod / 5,
                           displayPeriod / 5 };
local weightInit       = math.sqrt(
                           (1/patchSize[1])
                         * (1/patchSize[1])
                         * (1/inputFeatures));

------------
-- Column --
------------

local pvParams = {
   column = {
         groupType                   = "HyPerCol";
         startTime                   = 0;
         dt                          = 1;
         progressInterval            = 10;
         randomSeed                  = 1234567890;
         nx                          = inputWidth;
         ny                          = inputHeight;
         nbatch                      = nbatch;
         checkpointWrite             = true;
         checkpointWriteTriggerMode  = "step";
         checkpointWriteStepInterval = checkpointPeriod;
         deleteOlderCheckpoints      = true;
         numCheckpointsKept          = 2;
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
         normalizeLuminanceFlag = true;
         normalizeStdDev        = true;
         autoResizeFlag         = false;
         batchMethod            = "byFile";
         writeFrameToTimestamp  = true;
         resetToStartOnLoop     = false;
      }
   );


-- LCA Stack loop initial params --

local inputLayerName        = "Image";
local lcaPrefix             = "S";
local lcaIndex              = 1;
local inputLayerFeatures    = inputFeatures;
local inputLayerScale       = 1;
local basePhase             = 2;
local stackVThresh          = VThresh;
local lcaLayerName          = lcaPrefix .. lcaIndex;
local probeLayerName        = lcaLayerName;
local errorLayerName        = inputLayerName
                              .. "Recon"
                              .. lcaLayerName
                              .. "Error";
local reconLayerName        = inputLayerName
                              .. "Recon"
                              .. lcaLayerName;
local inputDCALayerScale    = nil;
local inputDCALayerName     = nil;
local inputDCALayerFeatures = nil;
local errorDCALayerName     = nil;
local reconDCALayerName     = nil;

for i_tier = 1, tiers do

   -- Param changes for next iteration --

   if i_tier > 1 then
      inputDCALayerScale    = inputLayerScale;
      inputLayerScale       = 1 / stride[i_tier-1];
      inputDCALayerName     = inputLayerName;
      inputLayerName        = lcaLayerName;
      basePhase             = basePhase + 3;
      stackVThresh          = stackVThresh * VThreshFactor;
      inputDCALayerFeatures = inputLayerFeatures
      inputLayerFeatures    = dictionarySize[i_tier-1];
      lcaIndex              = i_tier;
      lcaLayerName          = lcaPrefix .. lcaIndex;
      probeLayerName        = lcaLayerName;
      errorDCALayerName     = errorLayerName;
      errorLayerName        = inputLayerName .. "Recon" .. "Error";
      reconLayerName        = inputLayerName .. "Recon" .. lcaLayerName;
      reconDCALayerName     = inputDCALayerName .. "Recon" .. lcaLayerName;
   end


   -- LCA Stack Probes --

   local probePrefix = "";

   pv.addGroup(pvParams, errorLayerName .. "L2Probe", {
            groupType       = "L2NormProbe";
            targetLayer     = errorLayerName;
            message         = nil;
            textOutputFlag  = true;
            probeOutputFile = errorLayerName .. "L2Probe.txt";
            energyProbe     = probePrefix .. "EnergyProbe";
            coefficient     = 0.5;
            maskLayerName   = nil;
            exponent        = 2;
         }
      );

   
   pv.addGroup(pvParams, lcaLayerName .. "FirmThreshProbe", {
            groupType        = "FirmThresholdCostFnLCAProbe";
            targetLayer      = probeLayerName;
            message          = NULL;
            textOutputFlag   = true;
            probeOutputFile  = probeLayerName .. "FirmThreshProbe.txt";
            triggerLayerName = NULL;
            energyProbe      = "EnergyProbe";
            maskLayerName    = NULL;
         }
      );


   -- Error layer --

   pv.addGroup(pvParams, errorLayerName, {
            groupType        = "HyPerLayer";
            nxScale          = inputLayerScale;
            nyScale          = inputLayerScale;
            nf               = inputLayerFeatures;
            phase            = basePhase;
            InitVType        = "ZeroV";
            triggerLayerName = NULL;
            writeStep        = -1;
            initialWriteTime = -1;
         }
      );


   -- Recon layer --
   
   pv.addGroup(pvParams, reconLayerName, {
            groupType        = "HyPerLayer";
            nxScale          = inputLayerScale;
            nyScale          = inputLayerScale;
            nf               = inputLayerFeatures;
            phase            = basePhase + 2;
            InitVType        = "ZeroV";
            triggerLayerName = NULL;
            writeStep        = -1;
            initialWriteTime = -1;
         }
      );
  
   if i_tier > 1 then
      pv.addGroup(pvParams, reconDCALayerName, pvParams[reconLayerName], {
               nxScale = inputDCALayerScale;
               nyScale = inputDCALayerScale;
               nf      = inputDCALayerFeatures;
            }
         );
   end
   

   -- LCA layer --

   pv.addGroup(pvParams, lcaLayerName, {
            groupType              = "HyPerLCALayer";
            nxScale                = 1/stride[i_tier];
            nyScale                = 1/stride[i_tier];
            nf                     = dictionarySize[i_tier];
            phase                  = basePhase+1;
            mirrorBCflag           = false;
            valueBC                = 0;
            InitVType              = "ConstantV";
            valueV                 = stackVThresh;
            triggerLayerName       = NULL;
            writeStep              = displayPeriod;
            initialWriteTime       = displayPeriod;
            sparseLayer            = true;
            writeSparseValues      = true;
            updateGpu              = true;
            VThresh                = stackVThresh;
            AMin                   = AMin;
            AMax                   = AMax;
            AShift                 = 0;
            VWidth                 = stackVThresh
                                   * (tiers - i_tier + 1);
            clearGSynInterval      = 0;
            timeConstantTau        = timeConstantTau[i_tier];
            selfInteract           = true;
            adaptiveTimeScaleProbe = probePrefix .. "AdaptiveTimeScales";
         }
      );


   -- LCA Stack Connections --

   pv.addGroup(pvParams, inputLayerName .. "To" .. errorLayerName, {
            groupType     = "IdentConn";
            preLayerName  = inputLayerName;
            postLayerName = errorLayerName;
            channelCode   = 0;
         }
      );

   pv.addGroup(pvParams, reconLayerName .. "To" .. errorLayerName, {
            groupType     = "IdentConn";
            preLayerName  = reconLayerName;
            postLayerName = errorLayerName;
            channelCode   = 1;
         }
      );

   if i_tier > 1 then
      pv.addGroup(pvParams, reconDCALayerName .. "To" .. errorDCALayerName, {
               groupType     = "IdentConn";
               preLayerName  = reconDCALayerName;
               postLayerName = errorDCALayerName;
               channelCode   = 1;
            }
         );
   end

   pv.addGroup(pvParams, errorLayerName .. "To" .. lcaLayerName, {
		  groupType                     = "TransposeConn";
		  preLayerName                  = errorLayerName;
		  postLayerName                 = lcaLayerName;
		  channelCode                   = 0;
		  receiveGpu                    = useGpu;
		  updateGSynFromPostPerspective = true;
		  pvpatchAccumulateType         = "convolve";
		  writeStep                     = -1;
		  originalConnName              = lcaLayerName
                                                  .. "To"
                                                  .. errorLayerName;
	       }
            );

   if i_tier > 1 then
      pv.addGroup(pvParams,
                  errorDCALayerName .. "To" .. lcaLayerName,
		  pvParams[errorLayerName .. "To" .. lcaLayerName], {
		     preLayerName     = errorDCALayerName;
		     originalConnName = lcaLayerName
                                        .. "To"
                                        .. errorDCALayerName;
		  }
               );
   end
   
   pv.addGroup(pvParams, lcaLayerName .. "To" .. reconLayerName, {
		  groupType             = "CloneConn";
		  preLayerName          = lcaLayerName;
		  postLayerName         = reconLayerName;
		  channelCode           = 0;
		  writeStep             = -1;
		  delay                 = {0.000000};
		  pvpatchAccumulateType = "convolve";
		  originalConnName      = lcaLayerName
                                          ..
                                          "To"
                                          .. errorLayerName;
	       }
            );

   if i_tier > 1 then
      pv.addGroup(pvParams,
                  lcaLayerName .. "To" .. reconDCALayerName,
		  pvParams[lcaLayerName .. "To" .. reconLayerName], {
		     postLayerName    = reconDCALayerName;
		     originalConnName = lcaLayerName
                                        .. "To"
                                        .. errorDCALayerName;
		  }
               );
   end
   
   pv.addGroup(pvParams, lcaLayerName .. "To" .. errorLayerName, {
		  groupType                     = "MomentumConn";
		  preLayerName                  = lcaLayerName;
		  postLayerName                 = errorLayerName;
		  channelCode                   = -1;
		  delay                         = {0.000000};
		  plasticityFlag                = plasticity[i_tier];
		  sharedWeights                 = true;
		  weightInitType                = "UniformRandomWeight";
		  wMinInit                      = -weightInitRange[i_tier];
		  wMaxInit                      = weightInitRange[i_tier];
		  sparseFraction                = weightSparsity[i_tier];
		  triggerLayerName              = "Image";
		  triggerOffset                 = 0;
		  updateGSynFromPostPerspective = false;
                  pvpatchAccumulateType         = "convolve";
		  writeStep                     = -1;
		  initialWriteTime              = -1;
		  nxp                           = patchSize[i_tier];
		  nyp                           = patchSize[i_tier];
		  normalizeMethod               = "normalizeL2";
		  strength                      = 1;
		  normalizeArborsIndividually   = false;
		  normalizeOnInitialize         = true;
		  normalizeOnWeightUpdate       = true;
		  dWMax                         = learningRate[i_tier]; 
		  useMask                       = false;
		  momentumTau                   = momentumTau;
		  momentumMethod                = "viscosity";
		  momentumDecay                 = 0;
	       }
            );

   if (i_tier > 2) and (i_tier < tiers) then
      pv.addGroup(pvParams,
		  lcaLayerName .. "To" .. errorDCALayerName,
		  pvParams[lcaLayerName .. "To" .. errorLayerName], {
		     postLayerName = errorDCALayerName;
		     nxp           = patchSize[i_tier - 1]
                                   + (patchSize[i_tier] - 1)
                                   * (stride[i_tier - 1] / stride[i_tier - 2]);
		     nyp           = patchSize[i_tier - 1]
                                   + (patchSize[i_tier] - 1)
                                   * (stride[i_tier - 1] / stride[i_tier - 2]);
		  }
               );
      pvParams[lcaLayerName
               .. "To"
               .. errorDCALayerName].normalizeMethod = "normalizeGroup";
      pvParams[lcaLayerName
               .. "To"
               .. errorDCALayerName].normalizeGroupName = lcaLayerName
                                                          .. "To"
                                                          .. errorLayerName;
   elseif i_tier == 2 then
      pv.addGroup(pvParams,
		  lcaLayerName .. "To" .. errorDCALayerName,
		  pvParams[lcaLayerName .. "To" .. errorLayerName], {
		     postLayerName = errorDCALayerName;
		     nxp           = patchSize[i_tier - 1]
                                   + (patchSize[i_tier] - 1)
                                   * (stride[i_tier - 1]);
		     nyp           = patchSize[i_tier - 1]
                                   + (patchSize[i_tier] - 1)
                                   * (stride[i_tier - 1]);
		  }
               );
      pvParams[lcaLayerName
               .. "To"
               .. errorDCALayerName].normalizeMethod = "normalizeGroup";
      pvParams[lcaLayerName
               .. "To"
               .. errorDCALayerName].normalizeGroupName = lcaLayerName
                                                          .. "To"
                                                          .. errorLayerName;
   elseif i_tier == tiers then
      pv.addGroup(pvParams,
                  lcaLayerName .. "To" .. errorDCALayerName,
		  pvParams[lcaLayerName .. "To" .. errorLayerName], {
		     postLayerName = errorDCALayerName;
		     nxp           = inputWidth / (stride[i_tier - 2]);
		     nyp           = inputHeight / (stride[i_tier - 2]);
		  }
               );
      pvParams[lcaLayerName
               .. "To"
               .. errorDCALayerName].normalizeMethod = "normalizeGroup";
      pvParams[lcaLayerName
               .. "To"
               .. errorDCALayerName].normalizeGroupName = lcaLayerName
                                                          .. "To"
                                                          .. errorLayerName;
   end

   if i_tier > 1 then
      pv.addGroup(pvParams, errorLayerName .. "To" .. inputLayerName, {
		     groupType     = "IdentConn";
		     preLayerName  = errorLayerName;
		     postLayerName = inputLayerName;
		     channelCode   = 1;
	       }
            );
   end   

   pv.addGroup(pvParams, lcaLayerName .. "To" .. "Image" .. "_ATA", {
		  groupType                     = "HyPerConn";
		  preLayerName                  = lcaLayerName;
		  postLayerName                 = "Image";
		  channelCode                   = -1;
		  delay                         = {0.000000};
		  numAxonalArbors               = 1;
		  plasticityFlag                = true;
		  convertRateToSpikeCount       = false;
		  receiveGpu                    = false;
		  sharedWeights                 = true;
		  weightInitType                = "UniformRandomWeight";
		  wMinInit                      = 0;
		  wMaxInit                      = 0;
		  sparseFraction                = 0;
		  useListOfArborFiles           = false;
		  combineWeightFiles            = false;
		  initializeFromCheckpointFlag  = false;
		  triggerLayerName              = "Image";
		  triggerOffset                 = 0;
		  updateGSynFromPostPerspective = false;
		  pvpatchAccumulateType         = "convolve";
		  writeStep                     = -1;
		  initialWriteTime              = -1;
		  writeCompressedCheckpoints    = false;
		  selfFlag                      = false;
		  nxp                           = patchSizeATA[i_tier];
		  nyp                           = patchSizeATA[i_tier];
		  shrinkPatches                 = false;
		  normalizeMethod               = "none";
		  dWMax                         = 1;
	       }
            );

end  -- i_tier


------------
-- Probes --
------------

pv.addGroup(pvParams, "AdaptiveTimeScales", {
         groupType        = "AdaptiveTimeScaleProbe";
         targetName       = "EnergyProbe";
         message          = NULL;
         textOutputFlag   = true;
         probeOutputFile  = "AdaptiveTimeScales.txt";
         triggerLayerName = "Image";
         triggerOffset    = 0;
         baseMax          = 0.011;
         baseMin          = 0.01;
         tauFactor        = 0.025;
         growthFactor     = 0.025;
         writeTimeScales  = true;
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

return pvParams;
