classdef DMDMIC_Layer < nnet.layer.Layer    
        
    properties
    % fixed parameters
      amp       % scaling factor from measured images to images in [0 1]
      binR      % binR^2 normal pixels in a binnedPixel 
    end
        
    properties (Learnable)
    % Layer learnable parameters
      E         % Excitation patterns
      psf_ex    % Excitation psf
      psf_em    % Emission psf
      Illum     % Illumination profile
      mu_rd     % mean of read noise (added DC)
      sd_rd     % standard deviation of read noise
    end
    
    methods
        function layer = DMDMIC_Layer(name,psf_ex,psf_em,E,Illum,amp,binR,mu_rd,sd_rd)                         
          layer.Name    = name;            
          layer.E       = E;            
          layer.psf_ex  = dlarray(psf_ex);
          layer.psf_em  = dlarray(psf_em);

          layer.amp     = dlarray(amp);
          layer.binR    = binR;
          layer.mu_rd   = dlarray(mu_rd);
          layer.sd_rd   = dlarray(sd_rd);

          layer.Illum   = Illum;            
        end
        
        function Yhat = predict(layer,X)
          X         = abs(X);

          E_DMD     = getImplementableEX(layer,layer.E);              % DMD pattern           
          Iext_DMD  = layer.Illum .* E_DMD;                           % excitation image at the DMD            
          Iext_FP   = dltranspconv(Iext_DMD,layer.psf_ex.^2,0,...  
                                   'DataFormat','SSUB',...
                                   'Cropping','same');                % excitation image at the focal plane 
          Iem_FP    = X .* Iext_FP;                                   % emission image at the focal plane
          Iem_IP    = dltranspconv(Iem_FP,layer.psf_em.^2,0,...  
                                   'DataFormat','SSUB',...
                                   'Cropping','same');                % emission image at the image plane   

          % Binning (same as pooling)
          Y         = avgpool(Iem_IP,layer.binR,'Stride',layer.binR,...
                              'DataFormat','SSCB');                    % averaged image
          Y         = Y.*(layer.binR^2);                               % multiply by pixels in the bin for sum image  

%           % Add noise                   
%           N_norm    = normrnd(0,1,size(Y));
%           N_read    = normrnd(layer.mu_rd,layer.sd_rd,size(Y));
% 
%           Y_poiss   = sqrt(Y/layer.amp) .* N_norm + Y;                % add poisson noise (with gaussian approximatoin)
%           Y_rd      = Y_poiss + N_read/layer.amp;                     % add read noise
% 
%           Yhat      = Y_rd;

           Yhat =Y;% no noise mode 
        end

        function Aimp = getImplementableEX(layer,A)          
              Aimp = A;              
          % Aimp = 1./(1+exp(-A*10));    % activate while training for DMD               
        end                               
        
    end
end