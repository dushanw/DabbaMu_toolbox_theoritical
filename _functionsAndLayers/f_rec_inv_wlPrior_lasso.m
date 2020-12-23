% 20181107 by Dushan N. Wadduwage
% 20201223 edited by DNW to improve speed using GPU and fmin

function Xhat = f_rec_inv_wlPrior_lasso(pram,Ex,Yhat,gamma,wname)

tic
disp(['Preprocessing inputs | t = ' num2str(toc) '[s]'])
  
  % parameters
  if isempty(gamma) 
    gamma = 5e-5;       % the reconstruction is sensitive to the weight of the regularizer (for 64x64 gamma=3e-3 seems to work)
  end
  if isempty(wname) 
    wname       = 'db4';
  end
  
  % make double for sparse;
  Ex    = double(Ex);
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

  w = (sqrt(1./(y+1)));

  %% optimization over alpha 
disp(['Generating wavelet matrix | t = ' num2str(toc) '[s]'])
  Psy   = getWaveletmatrices(pram.Ny,pram.Nx,wname);
  
  AxPsy = single(full(A*Psy));
  Psy   = single(full(Psy));
  y     = single(y);
disp(['Lasso started | t = ' num2str(toc) '[s]'])
  alpha = lasso(AxPsy,y,'Options',statset('UseParallel',true)); 
disp(['Lasso done! | t = ' num2str(toc) '[s]'])
  x     = Psy*alpha;
  x(x<0)= 0;
  
  Xhat = reshape(x,pram.Ny,pram.Nx,size(x,2));
  
  % show reconstruction
  try
    implay(rescale(Xhat));  
%   imagesc(imtile(cat(3,rescale(Xhat), ...
%                        rescale(imresize(Yhat(:,:,1,:),[pram.Ny pram.Nx],'nearest'))...
%                 )));
%  axis image
  catch
    disp('cannot display results')
  end
end


function A_waverec2 = getWaveletmatrices(h,w,wname)
  X           = zeros(h,w);
  L           = wmaxlev(size(X),wname);
  [c s]       = wavedec2(X,2,wname);
  A_waverec2  = sparse(h*w,length(c));

  parfor i=1:length(c)
    c1{i}    = zeros(size(c));
    c1{i}(i) = 1;
    A_waverec2(:,i)=sparse(reshape(waverec2(c1{i},s,wname),[h*w,1])); 
  end
  done=1;
end

