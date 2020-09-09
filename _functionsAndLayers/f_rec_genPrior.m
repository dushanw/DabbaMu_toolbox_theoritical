
function [Xhat, opt_info] = f_rec_genPrior(pram,dlnet_fwd,dlnet_gen,Yhat,X0)

  alpha = dlarray(rand(1,1,pram.Ncompressed_gen,size(Yhat,4)),'SSCB');
  
  delta_alpha = 1e-3;
  for i=1:10000
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

    alpha                   = alpha - delta_alpha*dC_dAlpha;  
    fprintf('%d: Cost = %d\n',i,opt_info.track_cost(i))
  end
end

%% cost function
function [C, dC_dAlpha] = costFunc(dlnet_gen,dlnet_fwd,alpha,Yhat)
  Y = predict(dlnet_fwd,...
              predict(dlnet_gen,alpha)); 
  C  = mse(Y,Yhat);  

  [dC_dAlpha] = dlgradient(C,alpha);
end
