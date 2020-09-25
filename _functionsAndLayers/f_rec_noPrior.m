
function [Xhat, opt_info] = f_rec_noPrior(pram,dlnet_fwd,Yhat,X0)

  try 
    Xhat = f_rec_inv_noPrior(pram,dlnet_fwd,Yhat(:,:,:,1).extractdata,[]);
    Xhat = dlarray(single(cat(4,Xhat,Xhat)),'SSCB');
  catch
    Xhat = dlarray(single(rand(pram.Ny,pram.Nx,pram.Nc,size(Yhat,4))),'SSCB');
  end
  
  delta_X = 1e-1;
  
  averageGrad_Xhat   = [];
  averageSqGrad_Xhat = [];  
  for i=1:10000
    if rem(i-1,50)==0
       if ~isempty(X0)
         imagesc(imtile([rescale(X0.extractdata) rescale(Xhat.extractdata) rescale(imresize(Yhat(:,:,1,:).extractdata,[pram.Ny pram.Nx],'nearest'))]));axis image
       else
         imagesc(imtile([rescale(Xhat.extractdata) rescale(imresize(Yhat(:,:,1,:).extractdata,[pram.Ny pram.Nx],'nearest'))]));axis image
       end
       drawnow
    end

    [C, dC_dX]              = dlfeval(@costFunc, dlnet_fwd,Xhat,Yhat);
    opt_info.track_cost(i)  = C.extractdata;
    opt_info.trac_grad(i)   = max(dC_dX(:).extractdata); 

    [Xhat,averageGrad_Xhat,averageSqGrad_Xhat] = adamupdate(Xhat,dC_dX,averageGrad_Xhat,averageSqGrad_Xhat,i,...
                                                               0.01/log10(i+1));    
    % Xhat                    = Xhat - delta_X*dC_dX;
    fprintf('%d: Cost = %d\n',i,opt_info.track_cost(i))
  end
end

%% cost function
function [C, dC_dX] = costFunc(dlnet_fwd,Xhat,Yhat)
  Y = predict(dlnet_fwd, Xhat); 
  C  = mse(Y,Yhat);  

  [dC_dX] = dlgradient(C,Xhat);
end
