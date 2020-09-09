
clc; clear all; close all
addpath('./_functionsAndLayers/')
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

pram            = pram_init(); 

X               = f_get_dataset(pram);

%% train generator
lgraph_autoEnc  = f_gen_stdAutoEnc(pram);
trOptions       = f_set_training_options(pram,X.Val,X.Val);
net_autoEnc     = trainNetwork(X.Train,X.Train,lgraph_autoEnc,trOptions);
dlnet_gen       = f_get_gen(pram,net_autoEnc);

XhatTest        = predict(dlnet_gen,dlarray(10*rand(1,1,pram.Ncompressed_gen,100),'SSCB'));
imagesc(imtile([XhatTest.extractdata]))

%% image using the fwd model
pram.compression_fwd = 1;
pram.amp             = 1e10;                                % scaling factor from measured images to images in [0 1]
pram.mu_rd           = 0;
pram.sd_rd           = 0;
pram.compression_fwd = 4;
pram.Nt              = 64;
pram.binR            = floor(sqrt(pram.compression_fwd*pram.Nt));

dlnet_fwd       = f_gen_fwd(pram);
dlXTest         = dlarray(X.Test(:,:,:,randi(size(X.Test,4),1,2)),'SSCB');
Yhat            = predict(dlnet_fwd,dlXTest);
% subplot(1,2,1);imagesc(X.Test(:,:,1,1));axis image
% subplot(1,2,2);imagesc(Yhat(:,:,1,1).extractdata);axis image

%% reconstruct with gen prior 
[Xhat,opt_info] = f_rec_genPrior(pram,dlnet_fwd,dlnet_gen,Yhat,dlXTest);






