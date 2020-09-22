% 20200922 by Dushan N. Wadduwage
% Reconstruction by solving the liner system with a psudo inverse
% Yhat = AX => Xhat = A\Yhat

function Xhat = f_rec_inv_noPrior(pram,dlnet_fwd,Yhat,X0)
  
  % make double for sparse;
  Ex    = double(dlnet_fwd.Layers(2).E);
  Yhat  = double(Yhat);

  % make A
  i_vec = [1:pram.Ny*pram.Nx*pram.Nt]';
  j_vec = repmat(1:pram.Ny*pram.Nx,[1 pram.Nt])';
  s_vec = Ex(:);
  A = sparse(i_vec,j_vec,s_vec,pram.Ny*pram.Nx*pram.Nt,pram.Ny*pram.Nx);

  % solve for X using left devide (i.e. psudo inverse)
  Xhat = A\Yhat(:);  
  Xhat = reshape(Xhat,pram.Ny,pram.Nx);

  if ~isempty(X0)
    imagesc(imtile([rescale(X0.extractdata) ...
                    rescale(Xhat.extractdata) ...
                    rescale(imresize(Yhat(:,:,1,:).extractdata,[pram.Ny pram.Nx],'nearest'))...
                    ]));
    axis image       
  else
    imagesc(imtile([rescale(Xhat.extractdata) ...
                    rescale(imresize(Yhat(:,:,1,:).extractdata,[pram.Ny pram.Nx],'nearest'))...
                  ]));
    axis image
  end

end