
%% cts from main_testG_YYYYMMDD.m

clear track_cost trac_grad

XTest = XTest(:,:,:,1:2);
X0    = dlarray(XTest,'SSCB');  
Yhat  = predict(dlnet_fwd,X0);
alpha = dlarray(rand(1,1,Ncompressed,size(XTest,4)),'SSCB');

delta_alpha = 1e4;
for i=1:500
  if rem(i-1,50)==0
     Xhat    = predict(dlnet_gen,alpha);
     imagesc(imtile([rescale(X0.extractdata) rescale(Xhat.extractdata)]));axis image
     drawnow
  end

  [C, dC_dAlpha]  = dlfeval(@costFunc, dlnet_gen,dlnet_fwd,alpha,Yhat);
%  [C, dC_dAlpha]  = dlfeval(@costFunc, dlnet_gen,alpha,X0);
  
  track_cost(i)   = C.extractdata;
  trac_grad(i)    = max(dC_dAlpha(:).extractdata); 
  
  alpha           = alpha - delta_alpha*dC_dAlpha;  
  fprintf('%d: Cost = %d\n',i,track_cost(i))
end

% function [C, dC_dAlpha] = costFunc(dlnet_gen,alpha,X0)
%   Xhat  = predict(dlnet_gen,alpha); 
%   C     = mse(Xhat,X0);  
% 
%   [dC_dAlpha] = dlgradient(C,alpha,'RetainData',true);
% end

function [C, dC_dAlpha] = costFunc(dlnet_gen,dlnet_fwd,alpha,Yhat)
  Y = predict(dlnet_fwd,...
              predict(dlnet_gen,alpha)); 
  C  = mse(Y,Yhat);  

  [dC_dAlpha] = dlgradient(C,alpha);
end
