% 20200911 by Dushan N. Wadduwage
% Generator network for GAN (same network in the example: https://www.mathworks.com/help/deeplearning/ug/train-generative-adversarial-network.html)

function dlnetGenerator = f_gen_ganDeepGen(pram)

  numFilters      = pram.numFilters;
  numLatentInputs = pram.Ncompressed_gen;
  Nc              = pram.Nc; 
  
  filterSize      = 5;
  projectionSize  = [4 4 512];
  N_deeplayers    = 10;
  
  layersGenerator = [
      imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
      projectAndReshapeLayer(projectionSize,numLatentInputs,'proj');
      transposedConv2dLayer(filterSize,4*numFilters,'Name','tconv1')
      batchNormalizationLayer('Name','bnorm1')
      reluLayer('Name','relu1')
      transposedConv2dLayer(filterSize,2*numFilters,'Stride',2,'Cropping','same','Name','tconv2')
      batchNormalizationLayer('Name','bnorm2')
      reluLayer('Name','relu2')
      transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping','same','Name','tconv3')
      batchNormalizationLayer('Name','bnorm3')
      reluLayer('Name','relu3')
      %transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping','same','Name','tconv4')
      ];
    
  for i=1:N_deeplayers
      layersGenerator = [
          layersGenerator 
          convolution2dLayer(filterSize,numFilters,'Padding','same','Name',sprintf('conv%d',i));    
          batchNormalizationLayer('Name',sprintf('bn_c%d',i+3))
          reluLayer('Name',sprintf('relu_c%d',i+3))];
  end

  layersGenerator = [
          layersGenerator 
          convolution2dLayer(filterSize,1,'Padding','same','Name',sprintf('conv%d',i+1));    
          tanhLayer('Name','tanh')];

  lgraphGenerator = layerGraph(layersGenerator);
  dlnetGenerator  = dlnetwork(lgraphGenerator);

end