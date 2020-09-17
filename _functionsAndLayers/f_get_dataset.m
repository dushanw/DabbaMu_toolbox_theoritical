
function X = f_get_dataset(pram)
  
  switch pram.dataset
    case 'minist'
      load('./_Datasets/minist.mat');
      ITest   = [];
      
      XTrain  = XTrain - min(XTrain(:));
      XVal    = XVal - min(XVal(:));
      XTest  = XTest - min(XTest(:)); 
    case 'andrewCells_dapi_20x_maxProj'
      load('./_Datasets/andrewCells_dapi_20x_maxProj.mat')
      X0     = imresize(X_maxProj,0.5)+1;% +1 to get rid of the boder patches
      X0(find(X0(:)<0))=0;
      XTrain = [];
      ITest = X0(:,:,1)-1;
      for i=2:size(X0,3)
        for y_strt=1:pram.Ny/4:3*pram.Ny/4
          for x_strt=1:pram.Nx/4:3*pram.Nx/4
            X_temp = reshape(im2col(X0(y_strt:end,x_strt:end,i),[pram.Ny pram.Nx],'distinct'),...
                         pram.Ny, pram.Nx,1,[]);
            XTrain = cat(4,XTrain,X_temp);           
          end
        end
      end
      min_Xs  = min(min(XTrain,[],1),[],2);% filter boder regions          
      XTrain  = XTrain(:,:,1,find(min_Xs(:)>0));
      XTrain  = XTrain - 1;% substract the added 0 to get rid of the boder patches

      sum_Xs  = sum(sum(XTrain,1),2);% filter more empty regions
      XTrain  = XTrain(:,:,1,find(sum_Xs(:)>15));

      min_X   = min(XTrain(:));
      max_X   = max(XTrain(:));
      avg_X   = mean(XTrain(:));  

      XVal    = reshape(im2col(ITest(1:256,1:256),[pram.Ny pram.Nx],'distinct'),...
                         pram.Ny, pram.Nx,1,[]);
      XTest   = reshape(im2col(ITest(1:256,257:end),[pram.Ny pram.Nx],'distinct'),...
                         pram.Ny, pram.Nx,1,[]);
      ITest   = ITest(257:end,:);
                       
      XTrain  = 2*(XTrain)/(avg_X*6);% scale to [0 2] range on average
      XVal    = 2*(XVal)/(avg_X*6);
      XTest   = 2*(XTest)/(avg_X*6);
      ITest   = 2*(ITest)/(avg_X*6);
  end
  
  X.Train = imresize(XTrain,[pram.Ny pram.Nx]); 
  X.Val   = imresize(XVal,[pram.Ny pram.Nx]); 
  X.Test  = imresize(XTest,[pram.Ny pram.Nx]); 
  X.ITest = ITest;  
end