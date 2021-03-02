

%%
clc; clear all; close all
addpath(genpath('./_functionsAndLayers/'))
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

pram            = pram_init_sim(); 
pram.maxEpochs  = 1000;
pram.dropPeriod	= 200;
pram.avgCounts  = 10;

X               = f_get_dataset(pram);

%% initiate and train generator using AUTOENCODER
pram.encType    = 'fc_rnd_posNormWeights'; % {'fc_rnd','fc_had','fc_had_fixed','fc_rnd_posNormWeights'}
pram.encNoise   = 'poiss';
pram.excEnv     = 'multi-gpu';

lgraph_autoEnc  = f_gen_linAutoEnc(pram);
% lgraph_autoEnc  = f_gen_stdAutoEnc(pram);

A_start               = lgraph_autoEnc.Layers(2).Weights;
trOptions             = f_set_training_options(pram,X.Val,X.Val);
[net_autoEnc trinfo]  = trainNetwork(X.Train,X.Train,lgraph_autoEnc,trOptions);
A_trained             = net_autoEnc.Layers(2).Weights;

save(sprintf('~/Documents/tempData/net_autoEnc_%s_%s_%g.mat',pram.encType,pram.encNoise,pram.avgCounts),...
     'net_autoEnc','trinfo','A_start','A_trained')

%% test trained network
XhatTest_ae     = predict(net_autoEnc,X.Val);
imagesc(imtile([rescale(X.Val) rescale(XhatTest_ae)]));axis image;colorbar




