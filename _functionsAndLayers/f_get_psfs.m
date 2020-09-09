
function [psf_ex, psf_em] = f_get_psfs(pram)

  switch pram.psf_typ
    case 'gaussian'
      psf_ex  = fspecial('gaussian',15,.1);  % excitation psf
      psf_em  = fspecial('gaussian',15,.5);  % emission psf
  end
  
end