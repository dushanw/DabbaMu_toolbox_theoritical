%% observations (delete after addressing)
% 20200909: the genPrior network needs improvement. It seems to work better without gen prior.
% grad discent optimizer needs to be more sofisticated specissly the finite difference

%%
clc; clear all; close all
addpath('./_functionsAndLayers/')
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

pram            = pram_init(); 
X               = f_get_dataset(pram);

%% initiate and train generator
lgraph_autoEnc  = f_gen_stdAutoEnc(pram);
trOptions       = f_set_training_options(pram,X.Val,X.Val);
net_autoEnc     = trainNetwork(X.Train,X.Train,lgraph_autoEnc,trOptions);
dlnet_gen       = f_get_gen(pram,net_autoEnc);

% XhatTest        = predict(dlnet_gen,dlarray(10*rand(1,1,pram.Ncompressed_gen,100),'SSCB'));
% imagesc(imtile([XhatTest.extractdata]))

%% image using the fwd model
pram        = pram_init();
pram.amp    = 1e3; % scaling factor from measured images to images in [0 1]
pram.mu_rd  = 10;
pram.sd_rd  = 0;
pram.binR   = floor(sqrt(pram.compression_fwd*pram.Nt));

dlnet_fwd   = f_gen_fwd(pram);  % Noise module was commented 
dlXTest     = dlarray(X.Test(:,:,:,randi(size(X.Test,4),1,2)),'SSCB');

Yhat  = predict(dlnet_fwd,dlXTest);
Yhat  = Yhat.extractdata * pram.amp;
Yhat  = dlarray(poissrnd(Yhat)/pram.amp,'SSCB');% add poisson noise here seperately 

figure;
subplot(1,2,1);imagesc(X.Test(:,:,1,1));axis image
subplot(1,2,2);imagesc(Yhat(:,:,1,1).extractdata);axis image

%% reconstruct with gen prior 
[Xhat_genPrior,opt_info] = f_rec_genPrior(pram,dlnet_fwd,dlnet_gen,Yhat,dlXTest);
[Xhat_noPrior ,opt_info] = f_rec_noPrior(pram,dlnet_fwd,Yhat,dlXTest);

figure;
subplot(1,3,1);imagesc(dlXTest(:,:,1,1).extractdata);       axis image; axis off
subplot(1,3,2);imagesc(Xhat_genPrior(:,:,1,1).extractdata); axis image; axis off
subplot(1,3,3);imagesc(Xhat_noPrior(:,:,1,1).extractdata) ; axis image; axis off

savepath      = ['./__results/' date '/'];
mkdir(savepath)
fileNameStem  = sprintf('prior_vs_noPrior_1_DMDSim_Ny%d_Nx%d_Nt%d_comp%dx',pram.Ny,pram.Nx,pram.Nt,pram.compression_fwd)
saveas(gcf,[savepath fileNameStem '_fig.jpeg']);