
function X = f_get_dataset(pram)
  
  switch pram.dataset
    case 'minist'
      load('./_Datasets/minist.mat');
      ITest   = [];
      
      XTrain  = XTrain - min(XTrain(:));
      XVal    = XVal - min(XVal(:));
      XTest  = XTest - min(XTest(:));      
  end
  
  X.Train = imresize(XTrain,[pram.Ny pram.Nx]); 
  X.Val   = imresize(XVal,[pram.Ny pram.Nx]); 
  X.Test  = imresize(XTest,[pram.Ny pram.Nx]); 
  X.ITest = ITest;
  
end