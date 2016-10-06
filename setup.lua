#!/usr/bin/lua

-------------------
-- Initial Setup --
-------------------

-- Path to PetaVision binary
pathToBinary =
      os.getenv("HOME")
      .. "/workspace/OpenPV/build"
      .. "/tests/BasicSystemTest/Debug/BasicSystemTest";

-- Path to PetaVision source
pathToSource =
      os.getenv("HOME")
      .. "/workspace/OpenPV";

-- The network params file should use the values below.
-- The params file should *not* call pv.printConsole()
-- at the end. This script expects the network params
-- file to use the table params.
paramsFile = "networks/basic_lca.lua";
classifier = "networks/linear_classifier.lua";

-- Global Configuration
runVersion = 1;
runName    = "cifar_demo" .. runVersion;

displayPeriod   = 500;
columnWidth      = 32;
columnHeight     = 32;

inputTrainFiles = 50000;
inputTestFiles  = 10000;

unsupervisedEpochs = 2;
classifierEpochs   = 50;

debugParsing = false;

-- The layer names listed here will have thier inputPath and
-- displayPeriod updated automatically for train / test runs
inputLayerNames = {
      "Image"
   }; 
inputTrainLists = {
      "train.txt"
   };
inputTestLists  = {
      "test.txt"
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
