% 20200909 by Dushan N Wadduwage (DNW)
% 20201222 modified by DNW
% test for real data

%clc; clear all; close all
addpath(genpath('./_functionsAndLayers/'))
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

% setup cvx for gpu
% run('../_toolkits/cvx/cvx_startup.m')

pram  = pram_init(); 
% X   = f_get_dataset(pram);

gamma                 = 5e-4;
wname                 = 'db4';
savepath              = ['./__results/' date '_tfm_mouse_20201224/'];
mkdir(savepath)

%% set prams 
fileNameStem          = sprintf('%s_Ny%d_Nx%d_Nt%d',pram.pattern_typ,pram.Ny,pram.Nx,pram.Nt);
disp(fileNameStem)

%% read and preprocess data
[E Y_exp emConvSPSF Y_avg X_refs exp_names]= subf_readData(pram);

%% reconstruct and save results
clear Xhat_wlPr
for i=1:size(Y_exp,4)
  i
  Xhat_noPr(:,:,:,i)          = f_rec_inv_noPrior(pram,E,Y_exp(:,:,:,i),X_refs.Ywf0(:,:,:,i));  % no-prior 
% [Xhat_wlPr(:,:,:,i) FitInfo] = f_rec_inv_wlPrior_lasso(pram,E,Y_exp(:,:,:,i),emConvSPSF,gamma,wname,0.005);% wavelet-prior matlab lasso
 [Xhat_wlPr(:,:,:,i) FitInfo] = f_rec_inv_wlPrior_lasso(pram,E,Y_exp(:,:,:,i),emConvSPSF,gamma,wname,[]);% wavelet-prior matlab lasso
end
save([savepath 'reconstructed_' fileNameStem '.mat'],'Xhat_noPr','Xhat_wlPr')

for i=1:size(Y_exp,4)
  figure('units','normalized','outerposition',[0 0 1 1])          
  imagesc([rescale(Xhat_noPr(:,:,1,i))    rescale(Xhat_wlPr(:,:,1,i))   ;   ... 
           rescale(X_refs.Ywf0(:,:,1,i))  rescale(X_refs.Y_avg(:,:,1,i))]) ;axis image;colormap hot      
  %colormap jet  
  colormap gray  
  title('(1)Xhat-noPr  (2)Xhat-wlPr (3)Ywf0 (4)Yavg')
  set(gca,'fontsize',24)

  saveas(gcf,[savepath fileNameStem exp_names{i} '_fig.jpeg']);  
  close all
end


%% read data
function [E Y_exp emConvSPSF Y_avg X_refs exp_names] = subf_readData(pram)

  
  load('./_ExtPatternsets/dmd_exp_tfm_mouse_20201224.mat');
  dx          = pram.dx0*size(Data.Ex,1)/pram.Ny;
     
  load('./_PSFs/PSFs27-Dec-2020 04_21_23.mat');
  strt_ind    = (size(PSFs.emConvSPSF,1)+1+dx/PSFs.pram.dx)/2:-dx/PSFs.pram.dx:0;
  strt_ind    = ceil(strt_ind(end));
  emConvSPSF  = imresize(PSFs.emConvSPSF(strt_ind:end-strt_ind,strt_ind:end-strt_ind),PSFs.pram.dx/dx);

  
  E           = imresize(single(Data.Ex(:,:,1:pram.Nt)) ,[pram.Ny pram.Nx]);
  E           = E     -  mean(E    ,   3);
  E           = E     ./ max (E    ,[],3);
  
  fields      = fieldnames(Data);
  Y_exp_inds  = [2:6  12:16 22:25 30:32];
  Y_wf_inds   = [7:11 17:21 26:29 33:35];
  
  for i=1:length(Y_exp_inds)
    exp_names{i}    = fields{Y_exp_inds(i)};
    Y_exp(:,:,:,i)  = imresize(single(Data.(fields{Y_exp_inds(i)})(:,:,1:pram.Nt)),[pram.Ny pram.Nx]);
  end  
  Y_avg = mean(Y_exp,3);
  Y_avg = Y_avg ./ max(max(Y_avg,[],1),[],2);
  Y_exp = Y_exp -  mean(Y_exp,3);
  Y_exp = Y_exp ./ max(max(Y_exp,[],1),[],2);

  for i=1:length(Y_wf_inds)
%   exp_name_wf{i}  = fields{Y_wf_inds(i)};
    Ywf0(:,:,:,i)   = imresize(single(Data.(fields{Y_wf_inds(i)})(:,:,1:end)),[pram.Ny pram.Nx]);
  end
  Ywf0 = Ywf0 ./ max(max(Ywf0,[],1),[],2);

  X_refs.Ywf0   = Ywf0;      
  X_refs.Y_avg  = Y_avg;
  
end



