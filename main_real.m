% 20200909 by Dushan N Wadduwage
% test for real data

clc; clear all; close all
addpath('./_functionsAndLayers/')
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

pram            = pram_init(); 
X               = f_get_dataset(pram);

dlnet_fwd       = f_gen_fwd(pram);
dlXTest         = dlarray(X.Test(:,:,:,randi(size(X.Test,4),1,2)),'SSCB');
Yhat            = predict(dlnet_fwd,dlXTest);

load dmd_exp_USAF_20200813rsf1.mat
Y0    = imresize(DataSync(:,:,2:end),0.26);
Y0    = Y0(size(Y0,1)/2-pram.Ny/2+1:end,size(Y0,2)/2-pram.Nx/2+1:end,:);
Y0    = Y0(1:pram.Ny,1:pram.Nx,1:pram.Nt);   
Yacq  = Y0;
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


