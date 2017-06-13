function makeDirectories(runConfig)
   -- Create directories for each run and their results
   os.execute("mkdir -p " .. runConfig.runName);
   os.execute("mkdir -p " .. runConfig.runName .. "/dictionary");
   os.execute("mkdir -p " .. runConfig.runName .. "/groundtruth");
   os.execute("mkdir -p " .. runConfig.runName .. "/sparse");
   os.execute("mkdir -p " .. runConfig.runName .. "/sparse/train");
   os.execute("mkdir -p " .. runConfig.runName .. "/sparse/test");
   os.execute("mkdir -p " .. runConfig.runName .. "/weights");
   os.execute("mkdir -p " .. runConfig.paramsDir);
   os.execute("mkdir -p " .. runConfig.luaDir);
   os.execute("mkdir -p " .. runConfig.runsDir);
   os.execute("mkdir -p " .. runConfig.runsDir .. "learndictionary");
   os.execute("mkdir -p " .. runConfig.runsDir .. "writetrain");
   os.execute("mkdir -p " .. runConfig.runsDir .. "writetest");
   os.execute("mkdir -p " .. runConfig.runsDir .. "trainclassify");
   os.execute("mkdir -p " .. runConfig.runsDir .. "testclassify");
   os.execute("mkdir -p " .. runConfig.runsDir .. "scoretrain");
   os.execute("mkdir -p " .. runConfig.runName .. "/logs");
end

function backupScripts(runConfig, runParams)
   -- Copy the lua files being used into the project for future reference
   os.execute("cp " .. debug.getinfo(1).short_src .. " " .. runConfig.luaDir);
   os.execute("cp " .. runParams.paramsFile .. " " .. runConfig.luaDir);
   os.execute("cp " .. runParams.classifier .. " " .. runConfig.luaDir);
   os.execute("cp scripts/build.lua " .. runConfig.luaDir);
   os.execute("cp scripts/run_sparse.lua " .. runConfig.luaDir);
   os.execute("cp scripts/run_classify.lua " .. runConfig.luaDir);
   os.execute("cp scripts/run_analysis.lua " .. runConfig.luaDir);
end

function makeClassesTxt(runConfig, runParams)
   numCategories = 0;
   local classesFile = io.open(runConfig.runsDir .. "writetrain/classes.txt", "w");
   io.output(classesFile);
   for index, class in pairs(runParams.classes) do
      io.write(class .. "\n");
      numCategories = numCategories + 1;
   end
   io.close(classesFile);
   os.execute("cp "
            .. runConfig.runsDir .. "writetrain/classes.txt "
            .. runConfig.runsDir .. "writetest/classes.txt");
end

function sanitizeParamsFile(runConfig, paramsTable, suffix)
   local file = io.open(runConfig.paramsDir .. paramsTable.column.printParamsFilename, "w");
   io.output(file);
   pv.printConsole(paramsTable);
   io.close(file);


   print("Creating " .. suffix .. ".params");
   local command = 
         "cd " .. runConfig.runName .. "; "
         .. runConfig.pathToBinary .. " -p "
         .. "params/" .. paramsTable.column.printParamsFilename
         .. " -n -l /dev/null; "
         .. "cd -; cp "
         .. runConfig.runName .. "/runs/" .. suffix .. "/"
         .. paramsTable.column.printParamsFilename
         .. " " .. runConfig.paramsDir;
   os.execute(command);
end

function setNameAndLength(paramsTable, name, simTime)
   paramsTable.column.outputPath = "runs/" .. name;
   paramsTable.column.checkpointWriteDir = paramsTable.column.outputPath .. "/checkpoints";
   paramsTable.column.lastCheckpointDir = paramsTable.column.checkpointWriteDir .. "/last";
   paramsTable.column.printParamsFilename = name .. ".params";
   paramsTable.column.stopTime = simTime;
end

