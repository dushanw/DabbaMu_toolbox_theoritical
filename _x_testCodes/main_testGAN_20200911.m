% 20200911 by Dushan N. Wadduwage
% test a deep autoencoder 

cd('../')

clear all; close all; clc
addpath(genpath('./_functionsAndLayers/'))
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

pram      = pram_init();
X         = f_get_dataset(pram);
trOptions = f_set_training_options_gan(pram);

%Models.dlnetGenerator      = f_gen_ganDeepGen(pram);
Models.dlnetGenerator      = f_gen_stdGen(pram);
Models.dlnetDiscriminator  = f_gen_stdDisc(pram);
% analyzeNetwork(layerGraph(Models.dlnetGenerator))

% train new
[nets info] = f_train_gan(X.Train,X.Val,Models,trOptions);
dlnet_gen   = nets.dlnetGenerator;  

% retrain as need be
[nets info] = f_train_gan(X.Train,X.Val,nets,trOptions);
dlnet_gen   = nets.dlnetGenerator;  

XhatTest    = predict(dlnet_gen,gpuArray(dlarray(rand(1,1,pram.Ncompressed_gen,100,'single'),'SSCB')));
figure;imagesc(imtile([XhatTest.extractdata]));axis image


