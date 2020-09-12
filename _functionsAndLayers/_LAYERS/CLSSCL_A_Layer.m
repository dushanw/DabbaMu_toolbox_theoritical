classdef CLSSCL_A_Layer
        
  properties
  % fixed parameters
    Name
    amp       % scaling factor from measured images to images in [0 1]
    Nx 
    Ny
    Nt
    binR      % binR^2 normal pixels in a binnedPixel               
    A         % A for Y_lin = A X_lin
    At_aug    % inv(A_aug), A_aug is the A that can be derived by time mean substracted E and has better singular values
    mu_rd     % mean of read noise (added DC)
    sd_rd     % standard deviation of read noise
  end
    
  methods
    function layer = CLSSCL_A_Layer(name,E,amp,binR,mu_rd,sd_rd)                         
      layer.Name    = name;                  
      layer.Ny      = size(E,1);
      layer.Nx      = size(E,2);
      layer.Nt      = size(E,3);
      layer.amp     = amp;
      layer.binR    = binR;
      layer.mu_rd   = mu_rd;
      layer.sd_rd   = sd_rd;
      
      layer.A       = getA(layer,E,binR);            
      layer.At_aug  = getAt_aug(layer,E,binR);
      
    end

    function A = getA(layer,E,n)
      A = [];      
      for i=1:size(E,3)
        temp  = im2col(E(:,:,i),[n n],'distinct');
        temp  = num2cell(temp',2);
%        A     = [A; sparse(blkdiag(temp{:}))];
        A     = [A; blkdiag(temp{:})];
      end      
    end      
        
    function At_aug = getAt_aug(layer,E,n)       
      E_aug = E - mean(E,3);                % with the augmentation apperently the matrix At_aug gets badly scaled!
      E_aug = E_aug/max(E_aug(:));          
      A_aug = [];      
      for i=1:size(E_aug,3)
        temp  = im2col(E_aug(:,:,i),[n n],'distinct');
        temp  = num2cell(temp',2);
        A_aug = [A_aug; blkdiag(temp{:})];
      end      
      At_aug  = inv(A_aug);
    end
    
    function Yhat = fwd(layer,X)
      X    = im2col(X,[layer.binR layer.binR],'distinct');
      X    = X(:);
      Yhat = layer.A * X;
      Yhat = poissrnd(Yhat*layer.amp)/layer.amp + ...                         
             normrnd(layer.mu_rd,layer.sd_rd,size(Yhat))/layer.amp;
    end
    
    function Xhat = inv(layer,Yhat)      
%      Xhat = layer.At_aug * (Yhat-mean(Yhat));  % A_aug is badly scaled!
      Xhat = layer.A\Yhat;

      Xhat = reshape(Xhat,[layer.binR*layer.binR layer.Ny*layer.Nx/(layer.binR*layer.binR)]);
      Xhat = col2im(Xhat,[layer.binR layer.binR],[layer.Ny layer.Nx],'distinct');
    end

  end
end

