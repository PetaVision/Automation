#!/usr/bin/lua

-------------------
-- Initial Setup --
-------------------

-- Path to PetaVision binary
pathToBinary =
      os.getenv("HOME")
      .. "/workspace/OpenPV/build"
      .. "/tests/BasicSystemTest/Release/BasicSystemTest";

-- Path to PetaVision source
pathToSource =
      os.getenv("HOME")
      .. "/workspace/OpenPV";

-- Global Configuration
globalVThresh = 0.0025;
runVersion    = 1;

for v = 1,10 do

   print("************************************************");
   print("*  STARTING SWEEP ITERATION " .. runVersion);
   print("************************************************");

   -- Threads / Rows / Columns for sparse coding
   numSparseThreads = 8;
   numSparseRows    = 2;
   numSparseCols    = 2;

   -- Threads / Rows / Columns for classifier
   numClassThreads  = 32;
   numClassRows     = 1;
   numClassCols     = 1;

   -- The network params file should use the values below.
   -- The params file should *not* call pv.printConsole()
   -- at the end. This script expects the network params
   -- file to use the table params.
   paramsFile = "networks/basic_lca.lua";
   classifier = "networks/linear_classifier.lua";

   runVersion    = v;
   runName       = "cifar_vthresh_sweep_" .. runVersion;

   globalVThresh = globalVThresh * 2;
   displayPeriod = 500;
   columnWidth   = 32;
   columnHeight  = 32;

   inputTrainFiles = 50000;
   inputTestFiles  = 10000;

   unsupervisedEpochs = 1;
   classifierEpochs   = 50;

   debugParsing = false;

   -- The layer names listed here will have thier inputPath and
   -- displayPeriod updated automatically for train / test runs
   inputLayerNames = {
         "Image"
      }; 
   inputTrainLists = {
         "/shared/cifar-10-batches-mat/mixed_cifar.txt"
      };
   inputTestLists  = {
         "/shared/cifar-10-batches-mat/test_batch_randorder.txt"
      };

   -- The layer names listed here will be written to disk and
   -- used as input to the classification stage
   layersToClassify = {
         "S1"
      };

   -- These are automatically filled in below
   layersToClassifyFeatures = {};
   layersToClassifyXScale   = {};
   layersToClassifyYScale   = {};
   plasticConns             = {};

   -- This requires only one input layer be specified, and
   -- for that layer to be an ImageLayer
   generateGroundTruth = true;

   -- If the above flag is true, a ground truth pvp will be
   -- automatically created using this list and
   -- FilenameParsingGroundTruthLayer on the input layer above.
   classes = {
         "/0/",
         "/1/",
         "/2/",
         "/3/",
         "/4/",
         "/5/",
         "/6/",
         "/7/",
         "/8/",
         "/9/"
      };

   -------------------------------------------------------------
   -- TODO: Allow specifying ground truth input paths as well --
   -------------------------------------------------------------

   dofile("build.lua");
end
