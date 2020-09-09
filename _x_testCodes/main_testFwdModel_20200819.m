
clc; clear all; close all
% addpath('./_functionsAndLayers/')
% addpath('./_Datasets/')

amp     = 1e6;
mu_rd   = 100;
sd_rd   = 10;

Nx      = 128;
Nc      = 1;
Nt      = 16;
binR    = 1;
sizeIn  = [Nx Nx];
psf_ex  = fspecial('gaussian',15,1);
psf_em  = fspecial('gaussian',15,2);
Illum   = ones(sizeIn);
E       = rand([sizeIn Nt])-.5;
name    = 'fwdModel';

%% test A_layer
% FWD_A   = A_Layer(name,sizeIn,Nt,psf_ex,psf_em,E,Illum,amp,binR,mu_rd,sd_rd)
% FWD_A0  = A_Layer(name,sizeIn,Nt,psf_ex,psf_ex,E,Illum,amp,binR,mu_rd,sd_rd)
% checkLayer(FWD_A,[sizeIn 1 8],'ObservationDimension',4);
% lgraph  = layerGraph(FWD_A);
% analyzeNetwork(lgraph)

%% Gen fwd model as a dlnetwork
layers_fwd  = [
                imageInputLayer([Nx Nx Nc],'Normalization','none','Name','fwd_in')
                A_Layer(name,sizeIn,Nt,psf_ex,psf_em,E,Illum,amp,binR,mu_rd,sd_rd)
               ];
lgraph_fwd  = layerGraph(layers_fwd);
dlnet_fwd   = dlnetwork(lgraph_fwd);

%% test forward model
X0 = single(imread('img_UASF_target.jpg'));
X0 = imresize(mean(X0,3),sizeIn);
X0 = X0/max(X0(:));
% imagesc(X0);axis image;colorbar

Y  = predict(dlnet_fwd, dlarray(X0,'SSCB'));
imagesc([rescale(X0) rescale(Y(:,:,1).extractdata)]);axis image;colorbar





