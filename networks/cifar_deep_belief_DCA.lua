package.path = package.path .. ";" .. os.getenv("HOME") .. "/openpv/parameterWrapper/?.lua"; local pv = require "PVModule";
--local subnets = require "PVSubnets";


local inputWidth       = 32;
local inputHeight      = 32;
local inputFeatures    = 3;

local tiers            = 4;
local patchSize        = {5,     6,     6,     8};  
local patchSizeATA     = {5,     10,    20,    32}; -- effective size on retina  
local stride           = {1,     2,     4,     32};
--local dictionarySize   = {32,    128,   512,  2048};
--local dictionarySize   = {16,    64,    256,  16384};
local dictionarySize   = {16,    32,    64,  512};
local weightInitRange  = {1.0,   1.0,   1.0,   1.0};
local weightSparsity   = {0.9,   0.9,   0.9,   0.9};
local plasticity       = {true,  true,  true,  true};
local learningRate     = {1.0,   1.0,  1.0,   1.0};
local imageRecon       = true;
local topDownFeedback  = false;
local probeMaxPoolInstead = false;
--local individualEnergy = true;
local nbatch           = 4;      --Batch size
local displayPeriod    = 500;   --Number of timesteps to find sparse approximation
local numEpochs        = 1;      --Number of times to run through dataset
local numImages        = 50000;  --Total number of images in dataset
local stopTime         = (numImages * displayPeriod * numEpochs) / nbatch;
local writeStep        = displayPeriod * 10;
local initialWriteTime = displayPeriod * 10;

local runName          = "deep_belief_DCA";
local runVersion       = 1;
local imageInputPath   = "/Volumes/mountData/CIFAR/mixed_cifar.txt";
local outputPath       = "/Volumes/mountData/CIFAR/" .. runName .. "/" .. "train" .. runVersion;
local checkpointPeriod = (displayPeriod * 10);

local plasticityFlag   = true;  --Determines if we are learning weights or holding them constant
local momentumTau      = 200;

local VThreshFactor    = 1.0;
local VThresh          = 0.1;
local AMin             = 0;
local AMax             = infinity;
local timeConstantTau = { 50, 50, 50, 50 };

local weightInit       = math.sqrt((1/patchSize[1])*(1/patchSize[1])*(1/inputFeatures));

-- Base table variable to store
local pvParams = {
   column = {
      groupType = "HyPerCol";
      startTime                           = 0;
      dt                                  = 1;
      stopTime                            = stopTime;
      progressInterval                    = 10;
      writeProgressToErr                  = true;
      verifyWrites                        = false;
      outputPath                          = outputPath;
      printParamsFilename                 = "CIFAR" .. "_" .. runName .. "_" .. runVersion .. ".params";
      randomSeed                          = 1234567890;
      nx                                  = inputWidth;
      ny                                  = inputHeight;
      nbatch                              = nbatch;
      filenamesContainLayerNames          = 2;
      filenamesContainConnectionNames     = 2;
      initializeFromCheckpointDir         = "";
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
	       baseMax                             = 0.011;
	       baseMin                             = 0.01;
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
local stackVThresh = VThresh;
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
      stackVThresh          = stackVThresh * VThreshFactor;
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
		  valueV                              = stackVThresh;
		  triggerLayerName                    = NULL;
		  writeStep                           = displayPeriod;
		  initialWriteTime                    = displayPeriod;
		  sparseLayer                         = true;
		  writeSparseValues                   = true;
		  updateGpu                           = true;
		  dataType                            = nil;
		  VThresh                             = stackVThresh;
		  AMin                                = AMin;
		  AMax                                = AMax;
		  AShift                              = 0; --VThresh; --/ i;
		  VWidth                              = stackVThresh * (tiers - i_tier + 1); -- VThresh / 10.0;
		  clearGSynInterval                   = 0;
		  timeConstantTau                     = timeConstantTau[i_tier];
		  selfInteract                        = true;
		  adaptiveTimeScaleProbe              = probePrefix .. "AdaptiveTimeScales";
	       }
   )

   --LCA Stack Connections---------------------------------------------
   --------------------------------------------------------------------

   pv.addGroup(pvParams,
	       inputLayerName .. "To" .. errorLayerName,
	       {
		  groupType                           = "IdentConn";
		  preLayerName                        = inputLayerName;
		  postLayerName                       = errorLayerName;
		  channelCode                         = 0;
	       }
   )


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
		  plasticityFlag                      = plasticity[i_tier];
		  convertRateToSpikeCount             = false;
		  receiveGpu                          = false; -- non-sparse -> non-sparse
		  sharedWeights                       = true;
		  weightInitType                      = "UniformRandomWeight";
		  wMinInit                            = -weightInitRange[i_tier];
		  wMaxInit                            = weightInitRange[i_tier];
		  sparseFraction                      = weightSparsity[i_tier];
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
		  nxp                                 = patchSize[i_tier];
		  nyp                                 = patchSize[i_tier];
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
		  dWMax                               = learningRate[i_tier]; 
		  keepKernelsSynchronized             = true; -- Possibly irrelevant
		  useMask                             = false;
		  momentumTau                         = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
		  momentumMethod                      = "viscosity";
		  momentumDecay                       = 0;
	       }
   )

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
      --if i_tier > 2 then
      --	 pv.addGroup(pvParams,
      --		     errorDCALayerName .. "To" .. inputDCALayerName,
      --		     {
      --			groupType                           = "IdentConn";
      --			preLayerName                        = errorDCALayerName;
      --			postLayerName                       = inputDCALayerName;
      --			channelCode                         = 1;
      --		     }
      --	 )
      --end
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
		  --strength                            = 1;
		  --normalizeArborsIndividually         = false;
		  --normalizeOnInitialize               = true;
		  --normalizeOnWeightUpdate             = true;
		  --rMinX                               = 0;
		  --rMinY                               = 0;
		  --nonnegativeConstraintFlag           = false;
		  --normalize_cutoff                    = 0;
		  --normalizeFromPostPerspective        = false;
		  --minL2NormTolerated                  = 0;
		  dWMax                               = 1;
		  keepKernelsSynchronized             = true; -- Possibly irrelevant
		  useMask                             = false;
		  --momentumTau                         = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
		  --momentumMethod                      = "viscosity";
		  --momentumDecay                       = 0;
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
