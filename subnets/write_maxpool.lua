------------
-- Column --
------------

local pvWriteMaxpool = {
   column = {
      groupType                  = "HyPerCol";
      nx                         = runParams.columnWidth;
      ny                         = runParams.columnHeight;
      startTime                  = 0;
      dt                         = 1; 
      progressInterval           = 100;
      randomSeed                 = 1234567890;
      nbatch                     = runConfig.numClassBatches;
      checkpointWrite            = false;
      checkpointWriteTriggerMode = "step";
      deleteOlderCheckpoints     = false;
      errorOnNotANumber          = true;
   }
};

for index, layerName in pairs(runParams.layersToClassify) do
   pv.addGroup(pvWriteMaxpool, layerName, {
            groupType              = "PvpLayer";
            nxScale                = runParams.layersToClassifyXScale[layerName];
            nyScale                = runParams.layersToClassifyYScale[layerName];
            nf                     = runParams.layersToClassifyFeatures[layerName];
            phase                  = 0;
            displayPeriod          = 1;
            batchMethod            = "byFile";
            writeStep              = -1;
            initialWriteTime       = -1;
            resetToStartOnLoop     = false;
            normalizeLuminanceFlag = false;
            normalizeStdDev        = false;
            InitVType              = "ZeroV";
         }
      );

   pv.addGroup(pvWriteMaxpool, layerName .. "MaxPool", {
            groupType          = "HyPerLayer";
            nxScale            = runParams.maxPoolX / runParams.columnWidth;
            nyScale            = runParams.maxPoolY / runParams.columnHeight;
            nf                 = runParams.layersToClassifyFeatures[layerName];
            phase              = 1;
            writeStep          = 1;
            initialWriteTime   = 1;
            resetToStartOnLoop = false;
            sparseLayer        = true;
            writeSparseValues  = true;
            VThresh            = -infinity;
            AMin               = -infinity;
            AMax               = infinity;
            AShift             = 0;
            InitVType          = "ZeroV";
         }
      );

   pv.addGroup(pvWriteMaxpool, layerName .. "To" .. layerName .. "MaxPool", {
            groupType             = "PoolingConn";
            channelCode           = 0;
            preLayerName          = layerName;
            postLayerName         = layerName .. "MaxPool";
            pvpatchAccumulateType = "maxpooling";
            receiveGpu            = runParams.useGpu;
            writeStep             = -1;
            nxp                   = 1;
            nyp                   = 1;
            nfp                   = runParams.layersToClassifyFeatures[layerName];
         }
      );
end

return pvWriteMaxpool;
