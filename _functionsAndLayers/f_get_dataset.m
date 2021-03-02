
function X = f_get_dataset(pram)
  
  switch pram.dataset
    case 'minist'
      load('./_Datasets/minist.mat');
      ITest   = [];
      
      XTrain  = XTrain - min(XTrain(:));
      XVal    = XVal - min(XVal(:));
      XTest   = XTest - min(XTest(:)); 

      XTrain  = XTrain/2;
      XVal    = XVal/2;
      XTest   = XTest/2;
      
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
                       
      XTrain  = (XTrain)/(avg_X*6);% scale to [0 1] range on average
      XVal    = (XVal)/(avg_X*6);
      XTest   = (XTest)/(avg_X*6);
      ITest   = (ITest)/(avg_X*6);

    case 'andrewCells_fociW3_63x_maxProj'
      load('./_Datasets/andrewCells_fociW3_63x_maxProj.mat')
      X0     = imresize(X_maxProj,0.25);
      X0(find(X0(:)<0))=0;
      
      ITest   = X0(:,:,1);

      XVal    = reshape(im2col(X0(:,:,2),[pram.Ny pram.Nx],'distinct'),...
                        pram.Ny, pram.Nx,1,[]);
      XTest   = reshape(im2col(X0(:,:,3),[pram.Ny pram.Nx],'distinct'),...
                        pram.Ny, pram.Nx,1,[]);

      X0      = X0(:,:,4:end)+1;                     % +1 to get rid of the boder patches
      XTrain  = zeros(pram.Ny, pram.Nx, pram.Nc, size(X0,3)*16*ceil(size(X0,1)/pram.Ny)*ceil(size(X0,2)/pram.Nx));
      t=1;
      for i=1:size(X0,3)
        i
        for y_strt=1:pram.Ny/4:3*pram.Ny/4+1
          for x_strt=1:pram.Nx/4:3*pram.Nx/4+1
            X_temp = reshape(im2col(X0(y_strt:end,x_strt:end,i),[pram.Ny pram.Nx],'distinct'),...
                         pram.Ny, pram.Nx,1,[]);
            XTrain(:,:,1,t:t+size(X_temp,4)-1) = X_temp;    
            t = t+size(X_temp,4);
          end
        end
      end
      min_Xs  = min(min(XTrain,[],1),[],2);% filter boder regions          
      XTrain  = XTrain(:,:,1,find(min_Xs(:)>0));
      XTrain  = XTrain - 1;% substract the added 0 to get rid of the boder patches

      mean_Xs  = mean(mean(XTrain,1),2);% filter more empty regions
      XTrain  = XTrain(:,:,1,find(mean_Xs(:)>0.014));% for 128 size blocks

      min_X   = min(XTrain(:));
      max_X   = max(XTrain(:));
      avg_X   = mean(XTrain(:));  
                       
      XTrain  = (XTrain)/(avg_X*6);% scale to [0 1] range on average
      XVal    = (XVal)/(avg_X*6);
      XTest   = (XTest)/(avg_X*6);
      ITest   = (ITest)/(avg_X*6);
  end
  
  X.Train = imresize(XTrain,[pram.Ny pram.Nx]); 
  X.Val   = imresize(XVal,[pram.Ny pram.Nx]); 
  X.Test  = imresize(XTest,[pram.Ny pram.Nx]); 
  X.ITest = ITest;  
end