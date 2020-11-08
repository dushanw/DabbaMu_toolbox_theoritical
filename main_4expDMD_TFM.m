% 20200909 by Dushan N Wadduwage
% test for real data

clc; clear all; close all
addpath(genpath('./_functionsAndLayers/'))
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

pram  = pram_init(); 
% X   = f_get_dataset(pram);

[dlnet_fwd Yhat]  = f_gen_fwd(pram);

Xhat_noPr = f_rec_inv_noPrior(pram,dlnet_fwd,Yhat,[]);
Xhat_wlPr = f_rec_inv_wlPrior(pram,dlnet_fwd,Yhat);      % wavelet prior
 
imagesc([rescale(Xhat_noPr) rescale(Xhat_wlPr)]);axis image;colormap hot


Yacq  = Yhat;
Yacq  = Yacq - min(Yacq(:));
Yacq  = Yacq./max(Yacq(:));
Yacq  = dlarray(cat(4,Yacq,Yacq));
Yacq  = avgpool(Yacq,pram.binR,'Stride',pram.binR,...
                                'DataFormat','SSCB');                    % averaged image

[Xhat_noPrior ,opt_info] = f_rec_noPrior(pram,dlnet_fwd,Yacq,[]);

subplot(1,3,1);imagesc(mean(Y0,3));       axis image
subplot(1,3,2);imagesc(Xhat_noPrior(:,:,1,1).extractdata,[0 0.1]) ; axis image
subplot(1,3,3);imagesc(Yacq(:,:,1,1).extractdata) ; axis image

savepath      = ['./__results/' date '/'];
mkdir(savepath)
fileNameStem  = sprintf('real_noPrior2_DMDSim_Ny%d_Nx%d_Nt%d_comp%dx',pram.Ny,pram.Nx,pram.Nt,pram.compression_fwd)
saveas(gcf,[savepath fileNameStem '_fig.jpeg']);


