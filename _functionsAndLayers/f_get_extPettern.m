
function [E Y_exp X_refs] = f_get_extPettern(pram)

  switch pram.pattern_typ
    case 'dmd_sim_rnd'
      E     = rand([pram.Ny pram.Nx pram.Nt]); % for DMDs
      Y_exp = [];
    case 'wgd_sim'
      load wgd_simMMI.mat
      E     = E0(size(E0,1)/2-pram.Ny/2+1:end,size(E0,2)/2-pram.Nx/2+1:end,:);
      E     = E(1:pram.Ny,1:pram.Nx,1:pram.Nt);      
      Y_exp = [];
    case 'dmd_exp'
      load dmd_exp_USAF_20200813rsf1.mat
      E0    = imresize(ExtSync(:,:,2:end),0.26);
      E     = E0(size(E0,1)/2-pram.Ny/2+1:end,size(E0,2)/2-pram.Nx/2+1:end,:);
      E     = E(1:pram.Ny,1:pram.Nx,1:pram.Nt);   
      % E   = E - min(E(:));
      E     = E./max(E(:));

      Y_exp = imresize(DataSync(:,:,2:end),0.26);
      Y_exp = Y_exp(size(Y_exp,1)/2-pram.Ny/2+1:end,size(Y_exp,2)/2-pram.Nx/2+1:end,:);
      Y_exp = Y_exp(1:pram.Ny,1:pram.Nx,1:pram.Nt);   
      % Y_exp = Y_exp - min(Y_exp(:));
      Y_exp = Y_exp./max(Y_exp(:));
    case 'dmd_exp_tfm'
      load dmd_exp_tfm_mouseBrain_20200903.mat      
      E     = imresize(Data.Ex(:,:,1:pram.Nt)     ,[pram.Ny pram.Nx]);
      Y_exp = imresize(Data.z_200um(:,:,1:pram.Nt),[pram.Ny pram.Nx]);        
    case 'dmd_exp_tfm_beads_3'
      load dmd_exp_tfm_beads_20200925.mat
      E     = imresize(single(Data.Ex_3    (:,:,2:pram.Nt+1)) ,[pram.Ny pram.Nx]);
      Y_exp = imresize(single(Data.beads2_3(:,:,2:pram.Nt+1)) ,[pram.Ny pram.Nx]);
      X0    = imresize(single(Data.beads2_wf0)                ,[pram.Ny pram.Nx]);
      Xwf   = imresize(single(Data.beads2_wf(:,:,2:pram.Nt+1)),[pram.Ny pram.Nx]);
      % normalize
      E     = E     -  mean(E    ,   3);
      E     = E     ./ max (E    ,[],3);
      
      Y_exp = Y_exp -  mean(Y_exp,   3);
      Y_exp = Y_exp ./ max (Y_exp(:)  ); 
      
      X0    = X0    ./ max (X0(:)     );      
      
      Xwf   = mean(Xwf,3);
      Xwf   = Xwf   ./ max (Xwf(:)    );      
      
      X_refs.X0   = X0;
      X_refs.Xwf  = Xwf;    
    case 'dmd_exp_tfm_beads_4'
      load dmd_exp_tfm_beads_20200925.mat
      E     = imresize(single(Data.Ex_4    (:,:,2:pram.Nt+1)) ,[pram.Ny pram.Nx]);
      Y_exp = imresize(single(Data.beads2_4(:,:,2:pram.Nt+1)) ,[pram.Ny pram.Nx]);
      X0    = imresize(single(Data.beads2_wf0)                ,[pram.Ny pram.Nx]);
      Xwf   = imresize(single(Data.beads2_wf(:,:,2:pram.Nt+1)),[pram.Ny pram.Nx]);
      % normalize
      E     = E     -  mean(E    ,   3);
      E     = E     ./ max (E    ,[],3);
      
      Y_exp = Y_exp -  mean(Y_exp,   3);
      Y_exp = Y_exp ./ max (Y_exp(:)  ); 
      
      X0    = X0    ./ max (X0(:)     );      
      
      Xwf   = mean(Xwf,3);
      Xwf   = Xwf   ./ max (Xwf(:)    );      
      
      X_refs.X0   = X0;
      X_refs.Xwf  = Xwf;    
    case 'dmd_exp_tfm_beads_8'
      load dmd_exp_tfm_beads_20200925.mat
      E     = imresize(single(Data.Ex_8    (:,:,2:pram.Nt+1)) ,[pram.Ny pram.Nx]);
      Y_exp = imresize(single(Data.beads2_8(:,:,2:pram.Nt+1)) ,[pram.Ny pram.Nx]);
      X0    = imresize(single(Data.beads2_wf0)                ,[pram.Ny pram.Nx]);
      Xwf   = imresize(single(Data.beads2_wf(:,:,2:pram.Nt+1)),[pram.Ny pram.Nx]);
      % normalize
      E     = E     -  mean(E    ,   3);
      E     = E     ./ max (E    ,[],3);
      
      Y_exp = Y_exp -  mean(Y_exp,   3);
      Y_exp = Y_exp ./ max (Y_exp(:)  ); 
      
      X0    = X0    ./ max (X0(:)     );      
      
      Xwf   = mean(Xwf,3);
      Xwf   = Xwf   ./ max (Xwf(:)    );      
      
      X_refs.X0   = X0;
      X_refs.Xwf  = Xwf;          
    case 'dmd_exp_tfm_mouse20201119'
      load('/Volumes/GoogleDrive/My Drive/_Data/DEEP-TFM/2020-11-19/dmd_exp_tfm_mouseBrain_20201119_reg.mat');
      E     = imresize(single(Data.Ex           (:,:,1:pram.Nt)) ,[pram.Ny pram.Nx]);
      Y_exp = imresize(single(Data.reg1_100um   (:,:,1:pram.Nt)) ,[pram.Ny pram.Nx]);      
      Xwf   = imresize(single(Data.reg1_100um_wf(:,:,1:end))     ,[pram.Ny pram.Nx]);
      
      E     = E     - 300;
      Y_exp = Y_exp - 300;
      Xwf   = Xwf   - 300;
      % normalize
      Y_avg = mean(Y_exp,3);
      Y_avg = Y_avg ./ max (Y_avg(:)  );
      
      E     = E     -  mean(E    ,   3);
      E     = E     ./ max (E    ,[],3);
      
      Y_exp = Y_exp -  mean(Y_exp,   3);
      Y_exp = Y_exp ./ max (Y_exp(:)  ); 
      
      Xwf   = mean(Xwf,3);
      Xwf   = Xwf   ./ max (Xwf(:)    );      
      
      X_refs.X0     = [];
      X_refs.Xwf    = Xwf;
      X_refs.Y_avg  = Y_avg;          
    case 'dmd_exp_tfm_beads_7sls_20201219'
      load('./_ExtPatternsets/dmd_exp_tfm_beads_7sls_20201219.mat');
      E     = imresize(single(Data.Ex           (:,:,1:pram.Nt)) ,[pram.Ny pram.Nx]);
      Y_exp = imresize(single(Data.beads1_7sls  (:,:,1:pram.Nt)) ,[pram.Ny pram.Nx]);      
      Xwf0  = imresize(single(Data.beads1_sf_wf0(:,:,1:end))     ,[pram.Ny pram.Nx]);
      
      % normalize
      Y_avg = mean(Y_exp,3);
      Y_avg = Y_avg ./ max (Y_avg(:)  );
      
      E     = E     -  mean(E    ,   3);
      E     = E     ./ max (E    ,[],3);
      
      Y_exp = Y_exp -  mean(Y_exp,   3);
      Y_exp = Y_exp ./ max (Y_exp(:)  ); 
      
      Xwf   = Xwf0  ./ max (Xwf0(:)   );
      
      X_refs.X0     = Xwf0;      
      X_refs.Y_avg  = Y_avg;
    case 'dmd_exp_tfm_mouse_20201222'
      load('./_ExtPatternsets/dmd_exp_tfm_mouse_20201222.mat');
      E     = imresize(single(Data.Ex1               (:,:,1:pram.Nt)) ,[pram.Ny pram.Nx]);
      Y_exp = imresize(single(Data.anml1_r1_300um    (:,:,1:pram.Nt)) ,[pram.Ny pram.Nx]);      
      Xwf0  = imresize(single(Data.anml1_r1_300um_wf0(:,:,1:end))     ,[pram.Ny pram.Nx]);
      
      % normalize
      Y_avg = mean(Y_exp,3);
      Y_avg = Y_avg ./ max (Y_avg(:)  );
      
      E     = E     -  mean(E    ,   3);
      E     = E     ./ max (E    ,[],3);
      
      Y_exp = Y_exp -  mean(Y_exp,   3);
      Y_exp = Y_exp ./ max (Y_exp(:)  ); 
      
      Xwf   = Xwf0  ./ max (Xwf0(:)   );
      
      X_refs.X0     = Xwf0;      
      X_refs.Y_avg  = Y_avg;
    case 'dmd_exp_tfm_mouse_20201224'
      load('./_ExtPatternsets/dmd_exp_tfm_mouse_20201224.mat');
      E     = imresize(single(Data.Ex                (:,:,1:pram.Nt)) ,[pram.Ny pram.Nx]);
      Y_exp = imresize(single(Data.anml2_r2_400um    (:,:,1:pram.Nt)) ,[pram.Ny pram.Nx]);
      Xwf0  = imresize(single(Data.anml2_r2_400um_wf0(:,:,1:end))     ,[pram.Ny pram.Nx]);
      
      % normalize
      Y_avg = mean(Y_exp,3);
      Y_avg = Y_avg ./ max (Y_avg(:)  );
      
      E     = E     -  mean(E    ,   3);
      E     = E     ./ max (E    ,[],3);
      
      Y_exp = Y_exp -  mean(Y_exp,   3);
      Y_exp = Y_exp ./ max (Y_exp(:)  ); 
      
      Xwf   = Xwf0  ./ max (Xwf0(:)   );
      
      X_refs.X0     = Xwf0;      
      X_refs.Y_avg  = Y_avg;
  end

end