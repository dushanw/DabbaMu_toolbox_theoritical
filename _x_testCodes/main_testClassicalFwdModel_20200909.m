% 20200909 test classicA layer

% 20200909 observations: The direct inverse method doesnt seem to work well with noise.

clc;clear all;close all

pram  = pram_init();
X     = f_get_dataset(pram);
X0    = X.Test(:,:,:,randi(size(X.Test,4),1,2));
E     = f_get_extPettern(pram); 

name    = pram.mic_typ;
amp     = pram.amp; 
n       = pram.binR;
Nx      = pram.Nx;
Ny      = pram.Ny;        
Nt      = pram.Nt;
mu_rd   = pram.mu_rd;
sd_rd   = pram.sd_rd;
A_layer = CLSSCL_A_Layer(name,E,amp,n,mu_rd,sd_rd);

% X0      = reshape(1:32^2,[32 32])
Yhat    = A_layer.fwd(X0(:,:,1,1));
Xhat    = A_layer.inv(Yhat);

subplot(2,1,1);imagesc(X0(:,:,1,1));  axis image; colorbar
subplot(2,1,2);imagesc(Xhat);         axis image; colorbar

% bring Yhat back to the normal shape to compare with the dMu model
Yhat    = Yhat(1:pram.Ny/n*pram.Nx/n);
Yhat    = reshape(Yhat,[pram.Ny/n pram.Nx/n]);

% compare with dMu implementation to confirm accuray
dlnet_fwd = f_gen_fwd(pram);
dlX0      = dlarray(X0,'SSCB');
dlYhat    = predict(dlnet_fwd,dlX0);
Yhat_dMu  = dlYhat(:,:,1,1).extractdata;

subplot(2,2,1);imagesc(X0(:,:,1,1));  axis image; colorbar
subplot(2,2,2);imagesc(Yhat_dMu);     axis image; colorbar
subplot(2,2,3);imagesc(Yhat);         axis image; colorbar
subplot(2,2,4);imagesc(Yhat_dMu-Yhat);axis image; colorbar


