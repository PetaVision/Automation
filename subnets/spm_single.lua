
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

for i=1,runParams.inputFrames + runParams.temporalConvs do

   local startFrames = {};
   local skipFrames = {};
   for n=1,runConfig.numSparseBatches do
      startFrames[n] = (i + n) * runParams.inputFrameSkip;
      skipFrames[n] = runParams.inputFrameSkip-1;
   end

   pv.addGroup(pvParams, "Frame" .. i,  {
            groupType              = "ImageLayer";
            nxScale                = 1;
            nyScale                = 1;
            nf                     = runParams.inputFeatures;
            phase                  = 1;
            writeStep              = -1;
            initialWriteTime       = -1;
            offsetAnchor           = "cc";
            inverseFlag            = false;
            normalizeLuminanceFlag = false;
            normalizeStdDev        = false;
            autoResizeFlag         = true;
            rescaleMethod          = "bilinear";
            batchMethod            = "bySpecified";
            writeFrameToTimestamp  = true;
            resetToStartOnLoop     = false;
            start_frame_index      = startFrames;
            skip_frame_index       = skipFrames;
         }
      );
   
   pv.addGroup(pvParams, "Frame" .. i .. "ReconError", {
            groupType        = "HyPerLayer";
            nxScale          = 1;
            nyScale          = 1;
            nf               = runParams.inputFeatures;
            phase            = 2;
            writeStep        = -1;
            initialWriteTime = -1;
         }
      );

   pv.addGroup(pvParams, "Frame" .. i .. "Recon",  {
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
end

for t=1,runParams.temporalConvs do
   pv.addGroup(pvParams, "S1_" .. t , {
            groupType              = "HyPerLCALayer";
            nxScale                = 1/runParams.stride;
            nyScale                = 1/runParams.stride;
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
end
-----------------
-- Connections --
-----------------

for i=1,runParams.inputFrames + runParams.temporalConvs do

   pv.addGroup(pvParams, "Frame" .. i .. "ToFrame" .. i .."ReconError", {
            groupType     = "RescaleConn";
            preLayerName  = "Frame" .. i;
            postLayerName = "Frame" .. i .. "ReconError";
            channelCode   = 0;
            scale         = runParams.weightInit;
            initialWriteTime       = -1;
            writeStep              = -1;
         }
      );
   pv.addGroup(pvParams, "Frame" .. i .. "ReconToFrame" .. i .. "ReconError", {
            groupType     = "IdentConn";
            preLayerName  = "Frame" .. i .. "Recon";
            postLayerName = "Frame" .. i .. "ReconError";
            channelCode   = 1;
            initialWriteTime       = -1;
            writeStep              = -1;
         }
      );
end
 
for t=1,runParams.temporalConvs do
   for i=t,t+runParams.inputFrames do
      pv.addGroup(pvParams, "Frame" .. i .. "ReconErrorToS1_" .. t, {
               groupType                     = "TransposeConn";
               preLayerName                  = "Frame" .. i .. "ReconError";
               postLayerName                 = "S1_" .. t;
               channelCode                   = 0;
               receiveGpu                    = runParams.useGpu;
               updateGSynFromPostPerspective = true;
               pvpatchAccumulateType         = "convolve";
               writeStep                     = -1;
               originalConnName              = "S1_" .. t .. "ToFrame" .. i .. "ReconError";
            }
         );
     
      if t == 1 then
         pv.addGroup(pvParams, "S1_" .. t .. "ToFrame" .. i .. "ReconError", {
                  groupType               = "MomentumConn";
                  preLayerName            = "S1_" .. t;
                  postLayerName           = "Frame" .. i .. "ReconError";
                  channelCode             = -1;
                  plasticityFlag          = runParams.plasticityFlag;
                  sharedWeights           = true;
                  weightInitType          = "UniformRandomWeight";
                  wMinInit                = -1;
                  wMaxInit                = 1;
                  minNNZ                  = 1;
                  sparseFraction          = runParams.sparseFraction;
                  triggerLayerName        = "Frame" .. i;
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
      else
         pv.addGroup(pvParams, "S1_" .. t .. "ToFrame" .. i .. "ReconError", {
                  groupType               = "PlasticCloneConn";
                  preLayerName            = "S1_" .. t;
                  postLayerName           = "Frame" .. i .. "ReconError";
                  channelCode             = -1;
                  initialWriteTime        = -1;
                  writeStep               = -1;
                  originalConnName        = "S1_1ToFrame" .. (i-t+1) .. "ReconError";
               }
            );
      end 
      pv.addGroup(pvParams, "S1_" .. t .. "ToFrame" .. i .. "Recon", {
               groupType             = "CloneConn";
               preLayerName          = "S1_" .. t;
               postLayerName         = "Frame" .. i .. "Recon";
               channelCode           = 0;
               pvpatchAccumulateType = "convolve";
               originalConnName      = "S1_" .. t .. "ToFrame" .. i .. "ReconError";
               initialWriteTime       = -1;
               writeStep              = -1;
            }
         );
   end
end


------------
-- Probes --
------------

pv.addGroup(pvParams, "AdaptProbe", {
         groupType        = "KneeTimeScaleProbe";
         targetName       = "EnergyProbe";
         message          = NULL;
         textOutputFlag   = true;
         probeOutputFile  = "AdaptiveTimeScales.txt";
         triggerLayerName = "Frame1";
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

for i=1,runParams.inputFrames+runParams.temporalConvs do
   pv.addGroup(pvParams, "Frame" .. i .. "ReconErrorL2NormEnergyProbe", {
            groupType       = "L2NormProbe";
            targetLayer     = "Frame" .. i .. "ReconError";
            message         = nil;
            textOutputFlag  = true;
            probeOutputFile = "Frame" .. i .. "ReconErrorL2.txt";
            energyProbe     = "EnergyProbe";
            coefficient     = 0.5;
            maskLayerName   = nil;
            exponent        = 2;
         }
      );
end

for t=1,runParams.temporalConvs do
   pv.addGroup(pvParams, "S1_" .. t .. "L1NormEnergyProbe", {
            groupType       = "L1NormProbe";
            targetLayer     = "S1_" .. t;
            message         = nil;
            textOutputFlag  = true;
            probeOutputFile = "S1_" .. t .. "L1.txt";
            energyProbe     = "EnergyProbe";
            coefficient     = runParams.VThresh;
            maskLayerName   = nil;
         }
      );
end

-- Return our table. The file that calls this
-- one does the actual writing to disk.
return pvParams;
