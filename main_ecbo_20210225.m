

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
pram.encType    = 'fc_rnd_posNormWeights'; % {'fc_rnd','fc_had','fc_had_fixed','fc_rnd_posNormWeights','fc_rnd_posNormWeights_fixed'}
pram.encNoise   = 'poiss';
pram.excEnv     = 'multi-gpu';

lgraph_autoEnc  = f_gen_linAutoEnc(pram);

A_start               = lgraph_autoEnc.Layers(2).Weights;
trOptions             = f_set_training_options(pram,X.Val,X.Val);
[net_autoEnc trinfo]  = trainNetwork(X.Train,X.Train,lgraph_autoEnc,trOptions);
A_trained             = net_autoEnc.Layers(2).Weights;

saveName = sprintf('~/Documents/tempData/net_autoEnc_%s_%s_%g.mat',pram.encType,pram.encNoise,pram.avgCounts)
save(saveName,'net_autoEnc','trinfo','A_start','A_trained')

% scp harvard@10.245.73.7:~/Documents/tempData/net_autoEnc_fc_rnd_posNormWeights_poiss_10.mat net_autoEnc_fc_rnd_posNormWeights_poiss_10.mat


%% test trained network
XhatTest_ae     = predict(net_autoEnc,X.Val);
imagesc(imtile([rescale(X.Val) rescale(XhatTest_ae)]));axis image;colorbar

%% temp - compare validation loss
load('net_autoEnc_fc_rnd_posNormWeights_poiss_10.mat')
err_trainable_rnd_10  = trinfo.ValidationRMSE(find(~isnan(trinfo.ValidationRMSE)));
los_trainable_rnd_10  = trinfo.ValidationLoss(find(~isnan(trinfo.ValidationLoss)));
A_trainable_rnd_10    = [A_start; A_trained];

load('net_autoEnc_fc_rnd_posNormWeights_fixed_poiss_10.mat')
err_fixed_rnd_10      = trinfo.ValidationRMSE(find(~isnan(trinfo.ValidationRMSE)));
los_fixed_rnd_10      = trinfo.ValidationLoss(find(~isnan(trinfo.ValidationLoss)));
A_fixed_rnd_10        = [A_start; A_trained];

figure;imagesc([A_trainable_rnd_10;A_fixed_rnd_10])

plot(err_trainable_rnd_10);hold on
plot(err_fixed_rnd_10    );

plot(los_trainable_rnd_10);hold on
plot(los_fixed_rnd_10    );
