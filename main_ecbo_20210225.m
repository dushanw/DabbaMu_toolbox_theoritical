

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
pram.encType          = 'fc_rnd_posNormWeights'; % 'fc_rnd_posNormWeights_fixed'
pram.encNoise         = 'poiss';
pram.excEnv           = 'multi-gpu';
pram.compression_gen  = 8;
pram.compression_fwd  = 1/pram.Nt;
pram.Ncompressed_gen  = pram.Nx*pram.Ny/pram.compression_gen;            % dim non-linear feature space
pram.Ncompressed_fwd  = pram.Nx*pram.Ny/(pram.Nt*pram.compression_fwd);  % dim measurement space

lgraph_autoEnc  = f_gen_linAutoEnc(pram);

A_start               = lgraph_autoEnc.Layers(2).Weights;
trOptions             = f_set_training_options(pram,X.Val,X.Val);
[net_autoEnc trinfo]  = trainNetwork(X.Train,X.Train,lgraph_autoEnc,trOptions);
A_trained             = net_autoEnc.Layers(2).Weights;

saveDir   = '~/Documents/tempData/';
saveName  = sprintf('net_autoEnc_%s_comp-%g_%s-%g.mat',...
                                 pram.encType,...
                                         pram.compression_gen,...     
                                            pram.encNoise,...
                                               pram.avgCounts);
save([saveDir saveName],'net_autoEnc','trinfo','A_start','A_trained')

system(sprintf(['scp harvard@10.245.73.7:' saveDir saveName ' ' saveName])) 


%% test trained network
XhatTest_ae     = predict(net_autoEnc,X.Val);
imagesc(imtile([rescale(X.Val) rescale(XhatTest_ae)]));axis image;colorbar


