%% observations (delete after addressing)
% 20200909: the genPrior network needs improvement. It seems to work better without gen prior.
% grad discent optimizer needs to be more sofisticated specissly the finite difference

%%
clc; clear all; close all
addpath('./_functionsAndLayers/')
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

pram            = pram_init_sim(); 
X               = f_get_dataset(pram);

%% initiate and train generator using AUTOENCODER
lgraph_autoEnc  = f_gen_linAutoEnc(pram)
% lgraph_autoEnc  = f_gen_stdAutoEnc(pram);
% pram.excEnv     = 'multi-gpu'; 
pram.excEnv     = 'cpu'; 

trOptions       = f_set_training_options(pram,X.Val,X.Val);
net_autoEnc     = trainNetwork(X.Train,X.Train,lgraph_autoEnc,trOptions);
dlnet_gen       = f_get_gen(pram,net_autoEnc);

XhatTest_ae     = predict(net_autoEnc,X.Val);
imagesc(imtile([X.Val XhatTest_ae]));axis image;colorbar

XhatTest        = predict(dlnet_gen,dlarray(10*rand(1,1,pram.Ncompressed_gen,100),'SSCB'));
imagesc(imtile([XhatTest.extractdata]));axis image;colorbar

