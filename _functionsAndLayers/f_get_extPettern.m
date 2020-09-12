
function E = f_get_extPettern(pram)

  switch pram.pattern_typ
    case 'dmd_sim_rnd'
      E = rand([pram.Ny pram.Nx pram.Nt]); % for DMDs
    case 'wgd_sim'
      load wgd_simMMI.mat
      E = E0(size(E0,1)/2-pram.Ny/2+1:end,size(E0,2)/2-pram.Nx/2+1:end,:);
      E = E(1:pram.Ny,1:pram.Nx,1:pram.Nt);      
    case 'dmd_exp'
      load dmd_exp_USAF_20200813rsf1.mat
      E0  = imresize(ExtSync(:,:,2:end),0.26);
      E   = E0(size(E0,1)/2-pram.Ny/2+1:end,size(E0,2)/2-pram.Nx/2+1:end,:);
      E   = E(1:pram.Ny,1:pram.Nx,1:pram.Nt);   
      % E   = E - min(E(:));
      E   = E./max(E(:));
  end

end