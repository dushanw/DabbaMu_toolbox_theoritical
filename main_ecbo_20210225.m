%% observations (delete after addressing)
% 20200909: the genPrior network needs improvement. It seems to work better without gen prior.
% grad discent optimizer needs to be more sofisticated specissly the finite difference

%%
clc; clear all; close all
addpath('./_functionsAndLayers/')
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

pram            = pram_init_sim(); 
pram.maxEpochs  = 1000;
pram.dropPeriod	= 200;

X               = f_get_dataset(pram);

%% initiate and train generator using AUTOENCODER
pram.encType      = 'fc_rnd';
pram.encType      = 'fc_rnd_fixed';
pram.encType      = 'fc_had';
pram.encType      = 'fc_had_fixed';


lgraph_autoEnc  = f_gen_linAutoEnc(pram);
% lgraph_autoEnc  = f_gen_stdAutoEnc(pram);
pram.excEnv     = 'multi-gpu';

A_start         = lgraph_autoEnc.Layers(2).Weights;

trOptions       = f_set_training_options(pram,X.Val,X.Val);
net_autoEnc     = trainNetwork(X.Train,X.Train,lgraph_autoEnc,trOptions);

A_trained       = net_autoEnc.Layers(2).Weights;

dlnet_gen       = f_get_gen(pram,net_autoEnc);

XhatTest_ae     = predict(net_autoEnc,X.Val);
imagesc(imtile([X.Val XhatTest_ae]));axis image;colorbar

XhatTest        = predict(dlnet_gen,dlarray(10*rand(1,1,pram.Ncompressed_gen,100),'SSCB'));
imagesc(imtile([XhatTest.extractdata]));axis image;colorbar


