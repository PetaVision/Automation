#!/usr/bin/lua

------------
-- Params --
------------

local augmentation = 1;

runParams = {
   paramsFile = "subnets/dsga.lua";
   classifier = "subnets/maxpool_mlp.lua";

   runVersion    = 7;
   runPrefix     = "dsga_";

   
   displayPeriod      = 500;
   columnWidth        = 32;
   columnHeight       = 32;

   inputTrainFiles    = 50000 * augmentation;
   inputTestFiles     = 10000;

   unsupervisedEpochs = 1;
   classifierEpochs   = 50;

   useGpu             = true;

   -- LCA Params

   inputFeatures      = 3;
   plasticityFlag     = true;
   checkpointPeriod   = 500 * 10000;
   stride             = 1;
   patchSize          = 5;
   dictionarySize     = 32;
   VThresh            = 0.2;
   dWMax              = 0.0125;
   momentumTau        = 500;
   AMin               = 0;
   AMax               = infinity;
   AShift             = 0.2;
   VWidth             = 0;
   timeConstantTau    = 75;
   weightInit         = 1.0;
   sparseFraction     = 0.975;


   -- Classifier Params

   hiddenFeatures     = 64;
   maxPoolX           = 8;
   maxPoolY           = 8;
   hiddenXScale       = 0.5; -- This scale is relative to the maxPool dimensions
   hiddenYScale       = 0.5; 
   learningRate       = 0.001;
   learningRateDecay  = 0.9;
   numRateDecays      = 16; -- Learning rate will decay this many times over total run
   rateFactor         = 0.75; -- Hidden layers learn at this rate relative to the learning rate
   weightStd          = 0.01;
   hiddenPatch        = 2;
   connectionType     = "MomentumConn";
   momentumType       = "simple";
   momentum           = 0.25;
   decayFactor        = 0.01;-- momentum decay is factor * learningRate for each conn
   biasValue          = 1.0;
   inputDropout       = 33;
   hiddenDropout      = 66;
   normType           = "none";
   normStrength       = 0.1;
   normDW             = false;
   sharedWeights      = true;
   debugWriteStep     = -1;
   allHiddenLayer     = true; 
   allHiddenFeatures  = 128;
   
   enableSimpleClassifier = true;

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
         "S1", "S2", "S3"
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
   numCategories = 10;
};

fullRunName = runParams.runPrefix .. runParams.runVersion;

runConfig = {
   -- Threads / Rows / Columns for sparse coding
   numSparseThreads = 10;
   numSparseRows    = 1;
   numSparseCols    = 1;
   numSparseBatches = 5;

   -- Threads / Rows / Columns for classifier
   numClassThreads  = 5;
   numClassRows     = 1;
   numClassCols     = 1;
   numClassBatches  = 40;

   mpiBatchWidth = 5;

   runName   = fullRunName;
   paramsDir = fullRunName .. "/params/";
   runsDir   = fullRunName .. "/runs/";
   luaDir    = fullRunName .. "/lua/";

   -- Path to PetaVision binary
   pathToBinary =
         os.getenv("HOME")
         .. "/workspace/OpenPV/build"
         .. "/tests/BasicSystemTest/Release/BasicSystemTest";
   -- Path to PetaVision source
   pathToSource =
         os.getenv("HOME")
         .. "/workspace/OpenPV";
};


-------------------------------------------------------------
-- TODO: Allow specifying ground truth input paths as well --
-------------------------------------------------------------

dofile("scripts/build.lua");
