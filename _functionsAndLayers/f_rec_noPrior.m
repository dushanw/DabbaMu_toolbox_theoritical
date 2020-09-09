
function [Xhat, opt_info] = f_rec_noPrior(pram,dlnet_fwd,Yhat,X0)

  Xhat = dlarray(single(rand(pram.Ny,pram.Nx,pram.Nc,size(Yhat,4))),'SSCB');
  
  delta_X = 5e-3;
  for i=1:10000
    if rem(i-1,50)==0
       if ~isempty(X0)
         imagesc(imtile([rescale(X0.extractdata) rescale(Xhat.extractdata) rescale(imresize(Yhat(:,:,1,:).extractdata,[pram.Ny pram.Nx],'nearest'))]));axis image
       else
         imagesc(imtile([rescale(Xhat.extractdata)]));axis image
       end
       drawnow
    end

    [C, dC_dX]              = dlfeval(@costFunc, dlnet_fwd,Xhat,Yhat);
    opt_info.track_cost(i)  = C.extractdata;
    opt_info.trac_grad(i)   = max(dC_dX(:).extractdata); 

    Xhat                    = Xhat - delta_X*dC_dX;
    fprintf('%d: Cost = %d\n',i,opt_info.track_cost(i))
  end
end

%% cost function
function [C, dC_dX] = costFunc(dlnet_fwd,Xhat,Yhat)
  Y = predict(dlnet_fwd, Xhat); 
  C  = mse(Y,Yhat);  

  [dC_dX] = dlgradient(C,Xhat);
end
