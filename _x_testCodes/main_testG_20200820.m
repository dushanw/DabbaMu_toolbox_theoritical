
load('./_Datasets/minist.mat');     % loads XTrain, XVal, XTest

Nx = size(XTrain,2);
Ny = size(XTrain,1);
Nc = size(XTrain,3);

Compression = 32;
Ncompressed = Nx*Ny/Compression;

numFilters  = 64;
scale       = 0.01;                 % pramater of the leakyReLu

lgraph_autoEnc = f_stdAutoEnc(Nx,Nc,Ncompressed,numFilters,scale);

%% training and testing 
pram.maxEpochs          = 10;
pram.miniBatchSize      = 256;
pram.initLearningRate   = 1;
pram.learningRateFactor = .1;
pram.dropPeriod         = round(pram.maxEpochs/4);
pram.l2reg              = 0.0001;
pram.excEnv             = 'auto';   %'auto', 'gpu', 'multi-gpu'

trOptions               = set_training_options(pram,XVal,XVal);
net_autoEnc             = trainNetwork(XTrain,XTrain,lgraph_autoEnc,trOptions);

XhatTest                = net_autoEnc.predict(XTest);

imagesc(imtile([XhatTest XTest]))

%% takeout Gen part
lgraph_gen  = layerGraph(net_autoEnc)
layerNames  = {lgraph_gen.Layers(:).Name}
lgraph_gen  = removeLayers(lgraph_gen,layerNames(contains(layerNames,'enc')))
lgraph_gen  = addLayers(lgraph_gen,imageInputLayer([1 1 Ncompressed],'Normalization','none','Name','gen_in'));
lgraph_gen  = connectLayers(lgraph_gen,'gen_in','gen_tconv1');
lgraph_gen  = removeLayers(lgraph_gen,'out')

dlnet_gen   = dlnetwork(lgraph_gen);
XhatTest    = predict(dlnet_gen,dlarray(10*rand(1,1,Ncompressed,100),'SSCB'));

imagesc(imtile([XhatTest.extractdata]))

%% reconstruction using autoDiff
main_testFwdModel_20200819

main_testGenRec






