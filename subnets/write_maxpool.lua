------------
-- Column --
------------

local pvWriteMaxpool = {
   column = {
      groupType                  = "HyPerCol";
      nx                         = columnWidth;
      ny                         = columnHeight;
      startTime                  = 0;
      dt                         = 1; 
      progressInterval           = 100;
      randomSeed                 = 1234567890;
      nbatch                     = nbatch;
      checkpointWrite            = false;
      checkpointWriteTriggerMode = "step";
      deleteOlderCheckpoints     = false;
      errorOnNotANumber          = true;
   }
};

for index, layerName in pairs(layersToClassify) do
   pv.addGroup(pvWriteMaxpool, layerName, {
            groupType              = "PvpLayer";
            nxScale                = layersToClassifyXScale[layerName];
            nyScale                = layersToClassifyYScale[layerName];
            nf                     = layersToClassifyFeatures[layerName];
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
            nxScale            = maxPoolX / columnWidth;
            nyScale            = maxPoolY / columnHeight;
            nf                 = layersToClassifyFeatures[layerName];
            phase              = 1;
            writeStep          = 1;
            initialWriteTime   = 1;
            resetToStartOnLoop = false;
            VThresh            = -infinity;
            AMin               = -infinity;
            AMax               = infinity;
            AShift             = 0;
            probability        = inputDropout;
            InitVType          = "ZeroV";
         }
      );

   pv.addGroup(pvWriteMaxpool, layerName .. "To" .. layerName .. "MaxPool", {
            groupType             = "PoolingConn";
            channelCode           = 0;
            preLayerName          = layerName;
            postLayerName         = maxPoolLayerName;
            pvpatchAccumulateType = "maxpooling";
            receiveGpu            = useGpu;
            writeStep             = -1;
            nxp                   = 1;
            nyp                   = 1;
            nfp                   = layersToClassifyFeatures[layerName];
         }
      );
end

return pvWriteMaxpool;
