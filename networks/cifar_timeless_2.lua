#!/usr/bin/lua

------------
-- Params --
------------
-- Threads / Rows / Columns for sparse coding
numSparseThreads = 1;
numSparseRows    = 2;
numSparseCols    = 2;
numSparseBatches = 40;

-- Threads / Rows / Columns for classifier
numClassThreads  = 6;
numClassRows     = 1;
numClassCols     = 1;
numClassBatches  = 40;

mpiBatchWidth = 8;

paramsFile = "subnets/basic_lca.lua";
classifier = "subnets/maxpool_mlp.lua";

runVersion    = 2;
runName       = "cifar_timeless_" .. runVersion;

globalDictionarySize = 256;
globalVThresh = 0.075;
globalPatchSize = 7;

augmentation = 1;
displayPeriod = 150;
columnWidth   = 32;
columnHeight  = 32;

inputTrainFiles = 50000 * augmentation;
inputTestFiles  = 10000;

unsupervisedEpochs = 1;
classifierEpochs   = 50;

featureMult       = 1;
maxPoolX          = 8;
maxPoolY          = 8;
hiddenXScale      = 1/8; -- This scale is relative to the maxPool dimensions
hiddenYScale      = 1/8; 
nbatch            = numClassBatches;
learningRate      = 0.0001 * 0.25;
rateFactor        = 0.5; -- Hidden layers learn at this rate relative to the learning rate

-- Scale hidden feature count along with LCA feature count
hiddenFeatures    = 64 * featureMult;

--hiddenFeatures = 64;
useGpu            = true;
weightStd         = 0.01;
hiddenPatch       = 1;
connectionType    = "MomentumConn";
momentumType      = "simple";
momentum          = 0.5;
decayFactor       = 0.01; -- momentum decay is factor * learningRate for each conn
biasValue         = 1.0;
inputDropout      = 25;
hiddenDropout     = 50;
normType          = "none";
normStrength      = 0.1;
normDW            = false; 
sharedWeights     = true;
debugWriteStep    = -1;
allHiddenLayer    = false;
allHiddenFeatures = hiddenFeatures*2;

-- If this flag is true, the network trains an additional classifier
-- using the max pool layers as input to a softmax
if enableSimpleClassifier == nil then
   enableSimpleClassifier = true;
end

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

-- The network params file should use the values below.
-- The params file should *not* call pv.printConsole()
-- at the end. This script expects the network params
-- file to use the table params.
debugParsing = false;

-- The layer names listed here will have thier inputPath and
-- displayPeriod updated automatically for train / test runs
inputLayerNames = {
      "Image"
   }; 
inputLayerBatchMethods = {
      "byList"
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

dofile("scripts/build.lua");
