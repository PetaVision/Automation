package.path = package.path .. ";" .. os.getenv("HOME") .. "/openpv/parameterWrapper/?.lua"; local pv = require "PVModule";
--local subnets = require "PVSubnets";


local inputWidth       = 32;
local inputHeight      = 32;
local inputFeatures    = 3;

local tiers            = 4;
local patchSize        = {5,     6,     6,     8};  
local patchSizeATA     = {5,     10,    20,    32}; -- effective size on retina  
local stride           = {1,     2,     4,     32};
local dictionarySize   = {2*16,  2*32,  2*64,  2*512};
local weightInit       = 1.0; --math.sqrt((1/patchSize[1])*(1/patchSize[1])*(1/inputFeatures));
local weightInitRange  = {1.0,   1.0,   1.0,   1.0};
local weightSparsity   = {0.9,   0.9,   0.5,   0.5};
local plasticity       = {true,  true,  true,  true};
--local plasticity       = {false,  false,  false,  false};
local learningRate     = {1.0,   1.0,  1.0,   1.0};
for i_tier = 1, tiers do
   if not plasticity[i_tier] then
      learningRate[i_tier] = 0;
   end
end
local nbatch           = 4;      --Batch size
local displayPeriod    = 500;   --Number of timesteps to find sparse approximation
local numEpochs        = 1;      --Number of times to run through dataset
local numImages        = 60000;  --Total number of images in dataset
local stopTime         = (numImages * displayPeriod * numEpochs) / nbatch;
local writeStep        = displayPeriod * 10;
local initialWriteTime = displayPeriod * 10;

local runName          = "CIFAR10_DSGA_X2";
local runVersion       = 3;
local imageInputPath   = "/home/gkenyon/CIFAR10/mixed_cifar.txt";
local outputPath       = "/home/gkenyon/CIFAR10/" .. runName .. "/" .. "train" .. runVersion;
local initPath         = "/home/gkenyon/CIFAR10/" .. runName .. "/" .. "train" .. runVersion-1; -- 
local checkpointPeriod = (displayPeriod * 100);

local momentumTau      = 200;

--local VThreshLast      = 0.05;
--local VThresh          = 0.2;
local AMin             = 0;
local AMax             = infinity;
local timeConstantTau  = { 50,  50,  50,  50 };
local VThresh          = { 0.2; 0.2; 0.2*math.pow(.75,2); 0.2*math.pow(.75,3) };


-- Base table variable to store
local pvParams = {
   column = {
      groupType = "HyPerCol";
      startTime                           = 0;
      dt                                  = 1;
      stopTime                            = stopTime;
      progressInterval                    = checkpointPeriod;
      writeProgressToErr                  = true;
      verifyWrites                        = false;
      outputPath                          = outputPath;
      printParamsFilename                 = runName .. ".params";
      randomSeed                          = 1234567890;
      nx                                  = inputWidth;
      ny                                  = inputHeight;
      nbatch                              = nbatch;
      initializeFromCheckpointDir         = initPath .. "/Checkpoints/batchsweep_00/Checkpoint7500000"; -- 
      defaultInitializeFromCheckpointFlag = false;
      checkpointWrite                     = true;
      checkpointWriteDir                  = outputPath .. "/Checkpoints"; --The checkpoint output directory
      checkpointWriteTriggerMode          = "step";
      checkpointWriteStepInterval         = checkpointPeriod; --How often to checkpoint
      deleteOlderCheckpoints              = true;
      numCheckpointsKept                  = 2;
      suppressNonplasticCheckpoints       = false;
      writeTimescales                     = true;
      errorOnNotANumber                   = false;
   } 
}
if initPath then
   pvParams.column.initializeFromCheckpointDir = initPath .. "/Checkpoints/batchsweep_00/Checkpoint" .. stopTime;  
end

pv.addGroup(pvParams,
	    "AdaptiveTimeScales",
	    {
	       groupType = "AdaptiveTimeScaleProbe";
	       targetName                          = "EnergyProbe";
	       message                             = NULL;
	       textOutputFlag                      = true;
	       probeOutputFile                     = "AdaptiveTimeScales.txt";
	       triggerLayerName                    = "Image";
	       triggerOffset                       = 0;
	       baseMax                             = 0.0022; --0.011;
	       baseMin                             = 0.002; --0.01;
	       tauFactor                           = 0.025;
	       growthFactor                        = 0.025;
	       writeTimeScales                     = true;
	    }
)


--Layers----------------------------------------------------
------------------------------------------------------------

pv.addGroup(pvParams,
	    "Image", 
	    {
	       groupType                           = "ImageLayer";
	       nxScale                             = 1;
	       nyScale                             = 1;
	       nf                                  = 3;
	       phase                               = 0;
	       mirrorBCflag                        = true;
	       writeStep                           = writeStep;
	       initialWriteTime                    = initialWriteTime;
	       sparseLayer                         = false;
	       updateGpu                           = false;
	       dataType                            = nil;
	       inputPath                           = imageInputPath;
	       offsetAnchor                        = "cc";
	       offsetX                             = 0;
	       offsetY                             = 0;
	       writeImages                         = 0;
	       inverseFlag                         = false;
	       normalizeLuminanceFlag              = true;
	       normalizeStdDev                     = true;
	       jitterFlag                          = 0;
	       useInputBCflag                      = false;
	       padValue                            = 0;
	       autoResizeFlag                      = false; --true;
	       --aspectRatioAdjustment               = "pad";
	       --interpolationMethod                 = "bicubic";
	       displayPeriod                       = displayPeriod;
	       batchMethod                         = "byFile";
	       writeFrameToTimestamp               = true;
	       resetToStartOnLoop                  = false;
	    }
)

--LCA Stack loop--------------------------------------
------------------------------------------------------

local inputLayerName = "Image";
local lcaPrefix = "S";
local lcaIndex = 1;
local inputLayerFeatures = inputFeatures;
local inputLayerScale = 1;
local basePhase = 1;
--local stackVThresh = VThresh;
local errorWrite = -1 -- displayPeriod;

local lcaLayerName          = lcaPrefix .. lcaIndex;
local probeLayerName        = lcaLayerName;
local errorLayerName        = inputLayerName .. "Recon" .. lcaLayerName .. "Error";
local reconLayerName        = inputLayerName .. "Recon" .. lcaLayerName;
local inputDCALayerScale    = nil
local inputDCALayerName     = nil
local inputDCALayerFeatures = nil
local errorDCALayerName     = nil
local reconDCALayerName     = nil

for i_tier = 1, tiers do

   --LCA Stack Layers------------------------------------
   ------------------------------------------------------

   if i_tier > 1 then
      inputDCALayerScale    = inputLayerScale;
      inputLayerScale       = 1 / stride[i_tier-1];
      inputDCALayerName     = inputLayerName;
      inputLayerName        = lcaLayerName;
      basePhase             = basePhase + 3;
      --stackVThresh          = stackVThresh * VThreshFactor;
      inputDCALayerFeatures = inputLayerFeatures
      inputLayerFeatures    = dictionarySize[i_tier-1];
      lcaIndex              = i_tier
      lcaLayerName          = lcaPrefix .. lcaIndex;
      probeLayerName        = lcaLayerName;
      errorDCALayerName     = errorLayerName;
      errorLayerName        = inputLayerName .. "Recon" .. "Error";
      reconLayerName        = inputLayerName .. "Recon" .. lcaLayerName ;
      reconDCALayerName     = inputDCALayerName .. "Recon" .. lcaLayerName;
   end

   --------------------
   -- LCA Stack Probes
   --------------------
   local probePrefix = "";

   pv.addGroup(pvParams,
	       errorLayerName .. "L2Probe",
	       {
		  groupType                           = "L2NormProbe";
		  targetLayer                         = errorLayerName;
		  message                             = nil;
		  textOutputFlag                      = true;
		  probeOutputFile                     = errorLayerName .. "L2Probe.txt";
		  energyProbe                         = probePrefix .. "EnergyProbe";
		  coefficient                         = 0.5;
		  maskLayerName                       = nil;
		  exponent                            = 2;
	       }
   )

   
   pv.addGroup(pvParams,
	       lcaLayerName .. "FirmThreshProbe",
	       {
		  groupType                           = "FirmThresholdCostFnLCAProbe";
		  targetLayer                         = probeLayerName;
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = probeLayerName .. "FirmThreshProbe.txt";
		  triggerLayerName                    = NULL;
		  energyProbe                         = "EnergyProbe";
		  maskLayerName                       = NULL;
	       }
   )

   --Error layer
   pv.addGroup(pvParams,
	       errorLayerName,
	       {
		  groupType                           = "HyPerLayer";
		  nxScale                             = inputLayerScale;
		  nyScale                             = inputLayerScale;
		  nf                                  = inputLayerFeatures;
		  phase                               = basePhase;
		  mirrorBCflag                        = false;
		  valueBC                             = 0;
		  initializeFromCheckpointFlag        = false;
		  InitVType                           = "ZeroV";
		  triggerLayerName                    = NULL;
		  writeStep                           = writeStep;
		  initialWriteTime                    = initialWriteTime;
		  sparseLayer                         = false;
		  updateGpu                           = false;
		  dataType                            = nil;
	       }
   )


   --Recon layer
   pv.addGroup(pvParams,
	       reconLayerName,
	       {
		  groupType                           = "HyPerLayer";
		  nxScale                             = inputLayerScale;
		  nyScale                             = inputLayerScale;
		  nf                                  = inputLayerFeatures;
		  phase                               = basePhase + 2;
		  mirrorBCflag                        = false;
		  valueBC                             = 0;
		  initializeFromCheckpointFlag        = false;
		  InitVType                           = "ZeroV";
		  triggerLayerName                    = NULL;
		  writeStep                           = errorWrite;
		  initialWriteTime                    = errorWrite;
		  sparseLayer                         = false;
		  updateGpu                           = false;
		  dataType                            = nil;
	       }
   )
   if i_tier == 1 then
      pvParams[reconLayerName].writeStep        = writeStep;
      pvParams[reconLayerName].initialWriteTime = writeStep;
   end
   
   if i_tier > 1 then
      pv.addGroup(pvParams,
		  reconDCALayerName,
		  pvParams[reconLayerName],
		  {
		     nxScale                             = inputDCALayerScale;
		     nyScale                             = inputDCALayerScale;
		     nf                                  = inputDCALayerFeatures;
		  }
      )
      if i_tier == 2 then
	 pvParams[reconDCALayerName].writeStep        = writeStep;
	 pvParams[reconDCALayerName].initialWriteTime = writeStep;
      end
   end
   
   --LCA layer
   pv.addGroup(pvParams,
	       lcaLayerName,
	       {
		  groupType                           = "HyPerLCALayer";
		  nxScale                             = 1/stride[i_tier];
		  nyScale                             = 1/stride[i_tier];
		  nf                                  = dictionarySize[i_tier];
		  phase                               = basePhase+1;
		  mirrorBCflag                        = false;
		  valueBC                             = 0;
		  initializeFromCheckpointFlag        = false;
		  InitVType                           = "ConstantV";
		  valueV                              = VThresh[i_tier];
		  triggerLayerName                    = NULL;
		  writeStep                           = displayPeriod;
		  initialWriteTime                    = displayPeriod;
		  sparseLayer                         = true;
		  writeSparseValues                   = true;
		  updateGpu                           = true;
		  dataType                            = nil;
		  VThresh                             = VThresh[i_tier];
		  AMin                                = AMin;
		  AMax                                = AMax;
		  AShift                              = 0; --VThresh[i_tier];
		  VWidth                              = VThresh[i_tier] * (tiers - i_tier); -- VThresh / 10.0;
		  clearGSynInterval                   = 0;
		  timeConstantTau                     = timeConstantTau[i_tier];
		  selfInteract                        = true;
		  adaptiveTimeScaleProbe              = probePrefix .. "AdaptiveTimeScales";
	       }
   )
   if initPath and nbatch == 1 then
      pvParams[lcaLayerName].initializeFromCheckpointFlag = true;
   end

   if i_tier == tiers then
      local max_idle = 200;
      pv.addGroup(pvParams,
		  lcaLayerName .. "_" .. "integrator",
		  pvParams[lcaLayerName],
		  {
		     groupType                           = "LeakyIntegrator";
		     selfInteract                        = nil;
		     VWidth                              = 0.0; 
		     timeConstantTau                     = nil;
		     valueV                              = 0.0;
		     updateGpu                           = false;
		     phase                               = basePhase+2;
		  }
      )
      pvParams[lcaLayerName .. "_" .. "integrator"].triggerLayerName                    = "Image";
      pvParams[lcaLayerName .. "_" .. "integrator"].triggerBehavior                     = "updateOnlyOnTrigger";
      pvParams[lcaLayerName .. "_" .. "integrator"].triggerOffset                       = 0;
      pvParams[lcaLayerName .. "_" .. "integrator"].integrationTime                     = infinity;
      pv.addGroup(pvParams,
		  lcaLayerName .. "_" .. "constant",
		  {
		     groupType                           = "ConstantLayer";
		     nxScale                             = 1/stride[i_tier];
		     nyScale                             = 1/stride[i_tier];
		     nf                                  = dictionarySize[i_tier];
		     phase                               = 0;
		     mirrorBCflag                        = false;
		     valueBC                             = 0;
		     initializeFromCheckpointFlag        = false;
		     InitVType                           = "ConstantV";
		     valueV                              = VThresh[i_tier]/max_idle;
		     writeStep                           = -1;
		     sparseLayer                         = false;
		     updateGpu                           = false;
		     dataType                            = nil;
		     clearGSynInterval                   = 0;
		  }
      )
      pv.addGroup(pvParams,
		  lcaLayerName .. "_" .. "constant" .. "To" .. lcaLayerName .. "_" .. "integrator",
		  {
		     groupType                           = "IdentConn";
		     preLayerName                        = lcaLayerName .. "_" .. "constant";
		     postLayerName                       = lcaLayerName .. "_" .. "integrator";
		     channelCode                         = 0;
		  }
      )
      pv.addGroup(pvParams,
		  lcaLayerName .. "_" .. "integrator" .. "To" .. lcaLayerName,
		  {
		     groupType                           = "IdentConn";
		     preLayerName                        = lcaLayerName .. "_" .. "integrator";
		     postLayerName                       = lcaLayerName;
		     channelCode                         = 0;
		  }
      )
      pv.addGroup(pvParams,
		  lcaLayerName .. "To" .. lcaLayerName .. "_" .. "integrator",
		  {
		     groupType                           = "IdentConn";
		     preLayerName                        = lcaLayerName;
		     postLayerName                       = lcaLayerName .. "_" .. "integrator";
		     channelCode                         = 1;
		  }
      )
   end

   --LCA Stack Connections---------------------------------------------
   --------------------------------------------------------------------

   pv.addGroup(pvParams,
	       inputLayerName .. "To" .. errorLayerName,
	       {
		  groupType                           = "RescaleConn";
		  preLayerName                        = inputLayerName;
		  postLayerName                       = errorLayerName;
		  channelCode                         = 0;
		  scale                               = weightInit;
	       }
   )
   if i_tier > 1 then
      pvParams[inputLayerName .. "To" .. errorLayerName].groupType = "IdentConn";
      pvParams[inputLayerName .. "To" .. errorLayerName].scale     = nil;
   end


   pv.addGroup(pvParams,
	       reconLayerName .. "To" .. errorLayerName,
	       {
		  groupType                           = "IdentConn";
		  preLayerName                        = reconLayerName;
		  postLayerName                       = errorLayerName;
		  channelCode                         = 1;
	       }
   )

   if i_tier > 1 then
      pv.addGroup(pvParams,
		  reconDCALayerName .. "To" .. errorDCALayerName,
		  {
		     groupType                           = "IdentConn";
		     preLayerName                        = reconDCALayerName;
		     postLayerName                       = errorDCALayerName;
		     channelCode                         = 1;
		  }
      )
   end

   
   pv.addGroup(pvParams,
	       errorLayerName .. "To" .. lcaLayerName,
	       {
		  groupType                           = "TransposeConn";
		  preLayerName                        = errorLayerName;
		  postLayerName                       = lcaLayerName;
		  channelCode                         = 0;
		  delay                               = {0.000000};
		  convertRateToSpikeCount             = false;
		  receiveGpu                          = true;
		  updateGSynFromPostPerspective       = true;
		  pvpatchAccumulateType               = "convolve";
		  writeStep                           = -1;
		  writeCompressedCheckpoints          = false;
		  selfFlag                            = false;
		  gpuGroupIdx                         = -1;
		  originalConnName                    = lcaLayerName .. "To" .. errorLayerName;
	       }
   )

   if i_tier > 1 then
      pv.addGroup(pvParams,
		  errorDCALayerName .. "To" .. lcaLayerName,
		  pvParams[errorLayerName .. "To" .. lcaLayerName],
		  {
		     preLayerName                        = errorDCALayerName;
		     originalConnName                    = lcaLayerName .. "To" .. errorDCALayerName;
		  }
      )
   end
   
   pv.addGroup(pvParams,
	       lcaLayerName .. "To" .. reconLayerName,
	       {
		  groupType                           = "CloneConn";
		  preLayerName                        = lcaLayerName;
		  postLayerName                       = reconLayerName;
		  channelCode                         = 0;
		  writeStep                           = -1;
		  delay                               = {0.000000};
		  convertRateToSpikeCount             = false;
		  receiveGpu                          = false;
		  updateGSynFromPostPerspective       = false;
		  pvpatchAccumulateType               = "convolve";
		  writeCompressedCheckpoints          = false;
		  selfFlag                            = false;
		  originalConnName                    = lcaLayerName .. "To" .. errorLayerName;
	       }
   )

   if i_tier > 1 then
      pv.addGroup(pvParams,
		  lcaLayerName .. "To" .. reconDCALayerName,
		  pvParams[lcaLayerName .. "To" .. reconLayerName],
		  {
		     postLayerName                       = reconDCALayerName;
		     originalConnName                    = lcaLayerName .. "To" .. errorDCALayerName;
		  }
      )
   end
   
   pv.addGroup(pvParams,
	       lcaLayerName .. "To" .. errorLayerName,
	       {
		  groupType                           = "MomentumConn";
		  preLayerName                        = lcaLayerName;
		  postLayerName                       = errorLayerName;
		  channelCode                         = -1;
		  delay                               = {0.000000};
		  numAxonalArbors                     = 1;
		  convertRateToSpikeCount             = false;
		  receiveGpu                          = false; -- non-sparse -> non-sparse
		  sharedWeights                       = true;
		  useListOfArborFiles                 = false;
		  combineWeightFiles                  = false;
		  initializeFromCheckpointFlag        = false;
		  triggerLayerName                    = "Image";
		  triggerOffset                       = 0;
		  updateGSynFromPostPerspective       = false; -- Should be false from S1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
		  pvpatchAccumulateType               = "convolve";
		  writeStep                           = -1;
		  initialWriteTime                    = -1;
		  writeCompressedCheckpoints          = false;
		  selfFlag                            = false;
		  shrinkPatches                       = false;
		  normalizeMethod                     = "normalizeL2";
		  strength                            = 1;
		  normalizeArborsIndividually         = false;
		  normalizeOnInitialize               = true;
		  normalizeOnWeightUpdate             = true;
		  rMinX                               = 0;
		  rMinY                               = 0;
		  nonnegativeConstraintFlag           = false;
		  normalize_cutoff                    = 0;
		  normalizeFromPostPerspective        = false;
		  minL2NormTolerated                  = 0;
		  keepKernelsSynchronized             = true; -- Possibly irrelevant
		  useMask                             = false;
		  momentumTau                         = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
		  momentumMethod                      = "viscosity";
		  momentumDecay                       = 0;
		  nxp                                 = patchSize[i_tier];
		  nyp                                 = patchSize[i_tier];
		  plasticityFlag                      = plasticity[i_tier];
		  weightInitType                      = "UniformRandomWeight";
		  wMinInit                            = -weightInitRange[i_tier];
		  wMaxInit                            = weightInitRange[i_tier];
		  sparseFraction                      = weightSparsity[i_tier];
		  dWMax                               = learningRate[i_tier]; 
	       }
   )
   if initPath then
      pvParams[lcaLayerName .. "To" .. errorLayerName].initializeFromCheckpointFlag = true;
      pvParams[lcaLayerName .. "To" .. errorLayerName].weightInitType               = "FileWeight";
      if nbatch > 1 then
	 pvParams[lcaLayerName .. "To" .. errorLayerName].initWeightsFile              = initPath .. "/Checkpoints/batchsweep_00/Checkpoint" .. stopTime .. "/" ..  lcaLayerName .. "To" .. errorLayerName .. "_W.pvp";
      else
	 pvParams[lcaLayerName .. "To" .. errorLayerName].initWeightsFile              = initPath .. "/Checkpoints/Checkpoint" .. stopTime .. "/" ..  lcaLayerName .. "To" .. errorLayerName .. "_W.pvp";	 
      end
   end
   

   if (i_tier > 2) and (i_tier < tiers) then
      pv.addGroup(pvParams,
		  lcaLayerName .. "To" .. errorDCALayerName,
		  pvParams[lcaLayerName .. "To" .. errorLayerName],
		  {
		     postLayerName                       = errorDCALayerName;
		     nxp                                 = patchSize[i_tier-1] + (patchSize[i_tier]-1)*(stride[i_tier-1]/stride[i_tier-2]);
		     nyp                                 = patchSize[i_tier-1] + (patchSize[i_tier]-1)*(stride[i_tier-1]/stride[i_tier-2]);
		  }
      )
      pvParams[lcaLayerName .. "To" .. errorDCALayerName].normalizeMethod
	 = "normalizeGroup";
      pvParams[lcaLayerName .. "To" .. errorDCALayerName].normalizeGroupName                 
	 = lcaLayerName .. "To" .. errorLayerName;
   elseif i_tier == 2 then
      pv.addGroup(pvParams,
		  lcaLayerName .. "To" .. errorDCALayerName,
		  pvParams[lcaLayerName .. "To" .. errorLayerName],
		  {
		     postLayerName                       = errorDCALayerName;
		     nxp                                 = patchSize[i_tier-1] + (patchSize[i_tier]-1)*(stride[i_tier-1]);
		     nyp                                 = patchSize[i_tier-1] + (patchSize[i_tier]-1)*(stride[i_tier-1]);
		  }
      )
      pvParams[lcaLayerName .. "To" .. errorDCALayerName].normalizeMethod
	 = "normalizeGroup";
      pvParams[lcaLayerName .. "To" .. errorDCALayerName].normalizeGroupName                 
	 = lcaLayerName .. "To" .. errorLayerName;
   elseif i_tier == tiers then
      pv.addGroup(pvParams,
		  lcaLayerName .. "To" .. errorDCALayerName,
		  pvParams[lcaLayerName .. "To" .. errorLayerName],
		  {
		     postLayerName                       = errorDCALayerName;
		     nxp                                 = inputWidth/(stride[i_tier-2]);
		     nyp                                 = inputHeight/(stride[i_tier-2]);
		  }
      )
      pvParams[lcaLayerName .. "To" .. errorDCALayerName].normalizeMethod
	 = "normalizeGroup";
      pvParams[lcaLayerName .. "To" .. errorDCALayerName].normalizeGroupName                 
	 = lcaLayerName .. "To" .. errorLayerName;
   end   
   if i_tier > 1 and initPath then
      pvParams[lcaLayerName .. "To" .. errorDCALayerName].initializeFromCheckpointFlag = true;
      pvParams[lcaLayerName .. "To" .. errorDCALayerName].weightInitType               = "FileWeight";
      if nbatch > 1 then
	 pvParams[lcaLayerName .. "To" .. errorDCALayerName].initWeightsFile              = initPath .. "/Checkpoints/batchsweep_00/Checkpoint" .. stopTime .. "/" ..  lcaLayerName .. "To" .. errorDCALayerName .. "_W.pvp";
      else
	 pvParams[lcaLayerName .. "To" .. errorDCALayerName].initWeightsFile              = initPath .. "/Checkpoints/Checkpoint" .. stopTime .. "/" ..  lcaLayerName .. "To" .. errorDCALayerName .. "_W.pvp";
      end
   end


   if i_tier > 1 then
      pv.addGroup(pvParams,
		  errorLayerName .. "To" .. inputLayerName,
		  {
		     groupType                           = "IdentConn";
		     preLayerName                        = errorLayerName;
		     postLayerName                       = inputLayerName;
		     channelCode                         = 1;
	       }
      )
   end   


   pv.addGroup(pvParams,
	       lcaLayerName .. "To" .. "Image" .. "_ATA",
	       {
		  groupType                           = "HyPerConn";
		  preLayerName                        = lcaLayerName;
		  postLayerName                       = "Image";
		  channelCode                         = -1;
		  delay                               = {0.000000};
		  numAxonalArbors                     = 1;
		  plasticityFlag                      = true;
		  convertRateToSpikeCount             = false;
		  receiveGpu                          = false; -- non-sparse -> non-sparse
		  sharedWeights                       = true;
		  weightInitType                      = "UniformRandomWeight";
		  wMinInit                            = 0;
		  wMaxInit                            = 0;
		  sparseFraction                      = 0;
		  useListOfArborFiles                 = false;
		  combineWeightFiles                  = false;
		  initializeFromCheckpointFlag        = false;
		  triggerLayerName                    = "Image";
		  triggerOffset                       = 0;
		  updateGSynFromPostPerspective       = false; -- Should be false from S1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
		  pvpatchAccumulateType               = "convolve";
		  writeStep                           = -1;
		  initialWriteTime                    = -1;
		  writeCompressedCheckpoints          = false;
		  selfFlag                            = false;
		  nxp                                 = patchSizeATA[i_tier];
		  nyp                                 = patchSizeATA[i_tier];
		  shrinkPatches                       = false;
		  normalizeMethod                     = "none";
		  dWMax                               = 1;
		  keepKernelsSynchronized             = true; -- Possibly irrelevant
		  useMask                             = false;
	       }
   )

end  -- i_tier


--Probes------------------------------------------------------------
--------------------------------------------------------------------

pv.addGroup(pvParams,
	    "EnergyProbe",
	    {
	       groupType                           = "ColumnEnergyProbe";
	       message                             = nil;
	       textOutputFlag                      = true;
	       probeOutputFile                     = "EnergyProbe.txt";
	       triggerLayerName                    = nil;
	       energyProbe                         = nil;
	    }
)

-- Print out PetaVision approved parameter file to the console
pv.printConsole(pvParams)
