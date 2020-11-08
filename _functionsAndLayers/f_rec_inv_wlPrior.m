% 20181107 by Dushan N. Wadduwage

function Xhat = f_rec_inv_wlPrior(pram,dlnet_fwd,Yhat)

  % make double for sparse;
  Ex    = double(dlnet_fwd.Layers(2).E);
  Yhat  = double(Yhat);

  % make A
  i_vec = [1:pram.Ny*pram.Nx*pram.Nt]';
  j_vec = repmat(1:pram.Ny*pram.Nx,[1 pram.Nt])';
  s_vec = Ex(:);
  A = sparse(i_vec,j_vec,s_vec,pram.Ny*pram.Nx*pram.Nt,pram.Ny*pram.Nx);

%  y(find(y(:)<0))=0;  % no negative measurements 
  y = Yhat(:);
  y = y./max(y);

  y = A'*y;           % convert to a (Ny*Nx) by (Ny*Nx) system 
  A = A'*A;

  c = sum(A,1);
  A = A./max(c);
  c = c./max(c);

  n = size(A,2);

  gamma = 1e-3;       % the reconstruction is sensitive to the weight of the regularizer (for 64x64 gamma=3e-3 seems to work)
  Psy   = inv(getWaveletmatrices(pram.Ny,pram.Nx));

%    w = diag(sqrt(1./(y+1)));
  w = (sqrt(1./(y+1)));

  L1sum = norm(y(:),1);
  L2sum = sqrt(L1sum); % Target of least square estimation based on Poisson statistics
  eps = L2sum*1.5;

  cvx_begin 
      variable x(n)
      minimize(norm(w.*(A*x-y),2) + gamma*norm(Psy*x,1))        

      subject to
          x >= 0;
          %norm( A * x - y, 2 )<=eps;
  cvx_end

%   x(find(isnan(x)))=0;
%   x(find(x(:)<0))=0;

  Xhat = reshape(x,pram.Ny,pram.Nx);
  
  % show reconstruction
  imagesc(imtile([rescale(Xhat) ...
                  rescale(imresize(Yhat(:,:,1,:),[pram.Ny pram.Nx],'nearest'))...
                ]));
  axis image
  
end


function A_waverec2 = getWaveletmatrices(h,w)
  X = zeros(h,w);
  [c s]=wavedec2(X,2,'db1');  
  A_waverec2 = sparse(h*w,h*w);
  parfor i=1:h*w
%         c(:)=0;
%         c(i)=1;
%         Y_temp = waverec2(c,s,'haar');
%         A_waverec2(:,i)=sparse(reshape(Y_temp,[h*w,1])); 

    c1{i}    = zeros(size(c));
    c1{i}(i) = 1;
    A_waverec2(:,i)=sparse(reshape(waverec2(c1{i},s,'db1'),[h*w,1])); 
  end
end