% 20200909 by Dushan N Wadduwage (DNW)
% 20201222 modified by DNW
% test for real data

clc; clear all; close all
addpath(genpath('./_functionsAndLayers/'))
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

pram  = pram_init(); 
% X   = f_get_dataset(pram);

gamma             = 5e-4;
wname             = 'db4';

savepath      = ['./__results/' date '_test_wavelets/'];
mkdir(savepath)

%% set prams 
pram.compression_fwd  = 1/pram.Nt;
pram.Ncompressed_fwd  = pram.Nx*pram.Ny/(pram.Nt*pram.compression_fwd);  % dim measurement space
fileNameStem          = sprintf('%s_Ny%d_Nx%d_Nt%d',pram.pattern_typ,pram.Ny,pram.Nx,pram.Nt);

disp(fileNameStem)

%% start normal code here
[dlnet_fwd, E, Yhat, X_refs] = f_gen_fwd(pram);

Xhat_noPr       = f_rec_inv_noPrior       (pram,E,Yhat,X_refs.X0);        % no-prior 
Xhat_wlPr       = f_rec_inv_wlPrior       (pram,E,Yhat,gamma,wname);      % wavelet-prior
Xhat_wlPr_lasso = f_rec_inv_wlPrior_lasso (pram,E,Yhat,gamma,wname);      % wavelet-prior matlab lasso

figure('units','normalized','outerposition',[0 0 1 1])          
imagesc([rescale(Xhat_noPr)     rescale(Xhat_wlPr)   ;   ... 
         rescale(X_refs.X0)     rescale(X_refs.Y_avg)]) ;axis image;colormap hot      
title('(1)Xhat-noPr  (2)Xhat-wlPr (3)Xwf0 (4)Yavg')
set(gca,'fontsize',24)

%% save plot
saveas(gcf,[savepath fileNameStem 'reg1_100um_fig.jpeg']);  
close all

