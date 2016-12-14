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

-- Threads / Rows / Columns for sparse coding
numSparseThreads = 1;
numSparseRows    = 2;
numSparseCols    = 2;
numSparseBatches = 40;

-- Threads / Rows / Columns for classifier
numClassThreads  = 12;
numClassRows     = 1;
numClassCols     = 1;
numClassBatches  = 40;

mpiBatchWidth    = 4;

-- The network params file should use the values below.
-- The params file should *not* call pv.printConsole()
-- at the end. This script expects the network params
-- file to use the table params.
paramsFile = "subnets/basic_lca.lua";
classifier = "subnets/maxpool_mlp_res.lua";

-- Global Configuration
runVersion = 1;
runName    = "cifar_x10_" .. runVersion;

displayPeriod   = 250;
columnWidth      = 32;
columnHeight     = 32;

local augmentation = 10;
inputTrainFiles = 50000 * augmentation;
inputTestFiles  = 10000;

unsupervisedEpochs = 1; -- * (1.0 / augmentation); -- Uncomment this to only learn weights for one epoch
classifierEpochs   = 50;

debugParsing = false;

-- The layer names listed here will have thier inputPath and
-- displayPeriod updated automatically for train / test runs
inputLayerNames = {
      "Image"--"INPUT_LAYER_NAME_HERE"
   }; 
inputLayerBatchMethods = {
      "byList"
   };
inputTrainLists = {
      "/shared/cifar-10-batches-mat/mixed_cifar.txt"--"TRAIN_SET_LIST_HERE"
   };
inputTestLists  = {
      "/shared/cifar-10-batches-mat/test_batch_randorder.txt" --"TEST_SET_LIST_HERE"
   };

-- The layer names listed here will be written to disk and
-- used as input to the classification stage
layersToClassify = {
      "S1"--"CLASSIFICATION_LAYER_HERE"
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
