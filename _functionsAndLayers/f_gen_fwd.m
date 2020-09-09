
function dlnet_fwd = f_gen_fwd(pram)

  Illum             = ones([pram.Ny pram.Nx]);               % illumination 
  [psf_ex, psf_em]  = f_get_psfs(pram);           % psfs
  E                 = f_get_extPettern(pram);     % excitation pattern

  layers_fwd  = [
                  imageInputLayer([pram.Ny pram.Nx pram.Nc],'Normalization','none','Name','fwd_in')
                  DMDMIC_Layer(pram.mic_typ,...
                               psf_ex,psf_em,...
                               E,...
                               Illum,...
                               pram.amp,pram.binR,pram.mu_rd,pram.sd_rd)                               
                 ];
  lgraph_fwd  = layerGraph(layers_fwd);
  dlnet_fwd   = dlnetwork(lgraph_fwd);

end