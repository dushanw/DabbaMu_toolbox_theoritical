
function [Xhat, opt_info] = f_rec_genPrior(pram,dlnet_fwd,dlnet_gen,Yhat,X0)

  alpha = dlarray(rand(1,1,pram.Ncompressed_gen,size(Yhat,4),'single')-.5,'SSCB');
  alpha = gpuArray(alpha);
  
  delta_alpha0= 1e-3;
  Nitr        = 10000;
  delta_alpha = delta_alpha0*ones(1,Nitr);
  delta_alpha(Nitr/2:end) = delta_alpha(1)/10;
  
  averageGrad_alpha   = [];
  averageSqGrad_alpha = [];
  for i=1:Nitr
    if rem(i-1,50)==0
       Xhat    = predict(dlnet_gen,alpha);
       if ~isempty(X0)
         imagesc(imtile([rescale(X0.extractdata) rescale(Xhat.extractdata) rescale(imresize(Yhat(:,:,1,:).extractdata,[pram.Ny pram.Nx],'nearest'))]));axis image
       else
         imagesc(imtile([rescale(Xhat.extractdata)]));axis image
       end
       drawnow
    end

    [C, dC_dAlpha]          = dlfeval(@costFunc, dlnet_gen,dlnet_fwd,alpha,Yhat);
    opt_info.track_cost(i)  = C.extractdata;
    opt_info.trac_grad(i)   = max(dC_dAlpha(:).extractdata); 

    % try using adam update for update rather than simple gradient decent
    [alpha,averageGrad_alpha,averageSqGrad_alpha] = adamupdate(alpha,dC_dAlpha,averageGrad_alpha,averageSqGrad_alpha,i,...
                                                               0.01/log10(i+1));
    
%     alpha                   = alpha - delta_alpha(i)*dC_dAlpha;  
    fprintf('%d: Cost = %d\n',i,opt_info.track_cost(i))
  end
end

%% cost function
function [C, dC_dAlpha] = costFunc(dlnet_gen,dlnet_fwd,alpha,Yhat)
  Xest  = predict(dlnet_gen,alpha) + 1; % +1 to scale the [-1 1] to [0 2]
%  Xest  = sqrt(Xest.^2);
  Yest  = predict(dlnet_fwd,Xest);            
  C     = mse(Yest,Yhat);  

  [dC_dAlpha] = dlgradient(C,alpha);
end
