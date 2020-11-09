% 20200909 by Dushan N Wadduwage
% test for real data

clc; clear all; close all
addpath(genpath('./_functionsAndLayers/'))
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

pram  = pram_init(); 
% X   = f_get_dataset(pram);

% pattern_typ_list  = {'dmd_exp_tfm_beads_3', 'dmd_exp_tfm_beads_4','dmd_exp_tfm_beads_8'};
% Nx_list           = [64 128 256 326];
% Nt_list           = [64 128 245];

pattern_typ_list  = {'dmd_exp_tfm_beads_3', 'dmd_exp_tfm_beads_4','dmd_exp_tfm_beads_8'};
Nx_list           = [32 64];
Nt_list           = [64 128];

savepath      = ['./__results/' date '/'];
mkdir(savepath)

for ii = 1:length(pattern_typ_list)
  for jj = 1:length(Nx_list)
    for kk = 1:length(Nt_list)      
      %% set prams 
      pram.pattern_typ      = pattern_typ_list{ii};
      pram.Nx               = Nx_list(jj);
      pram.Ny               = Nx_list(jj);
      pram.Nt               = Nt_list(kk);
      pram.compression_fwd  = 1/pram.Nt;
      pram.Ncompressed_fwd  = pram.Nx*pram.Ny/(pram.Nt*pram.compression_fwd);  % dim measurement space
      fileNameStem          = sprintf('%s_Ny%d_Nx%d_Nt%d',pram.pattern_typ,pram.Ny,pram.Nx,pram.Nt);
      
      disp(fileNameStem)
      
      %% start normal code here
      [dlnet_fwd, Yhat, X_refs] = f_gen_fwd(pram);

      Xhat_noPr = f_rec_inv_noPrior(pram,dlnet_fwd,Yhat,X_refs.X0);
      Xhat_wlPr = f_rec_inv_wlPrior(pram,dlnet_fwd,Yhat);      % wavelet prior
      
      figure('units','normalized','outerposition',[0 0 1 1])          
      imagesc([rescale(Xhat_noPr) rescale(Xhat_wlPr) ;...
               rescale(X_refs.X0) rescale(X_refs.Xwf) ...
              ]);axis image;colormap hot      
      title('(1)Xhat-noPr  (2)Xhat-wlPr  (3)X0  (4)Xwf')
      set(gca,'fontsize',24)
      
      %% My own optimzation loop      
      % Yacq  = Yhat;
      % Yacq  = Yacq - min(Yacq(:));
      % Yacq  = Yacq./max(Yacq(:));
      % Yacq  = dlarray(cat(4,Yacq,Yacq));
      % Yacq  = avgpool(Yacq,pram.binR,'Stride',pram.binR,...
      %                                 'DataFormat','SSCB');                    % averaged image
      % 
      % dlX0  = dlarray(cat(4,X0,X0));
      % [Xhat_noPrior ,opt_info] = f_rec_noPrior(pram,dlnet_fwd,Yacq,dlX0);
      % 
      % subplot(1,3,1);imagesc(mean(Y0,3));       axis image
      % subplot(1,3,2);imagesc(Xhat_noPrior(:,:,1,1).extractdata,[0 0.1]) ; axis image
      % subplot(1,3,3);imagesc(Yacq(:,:,1,1).extractdata) ; axis image

      %% save plot
      saveas(gcf,[savepath fileNameStem '_fig.jpeg']);  
      close all
    end
  end
end

