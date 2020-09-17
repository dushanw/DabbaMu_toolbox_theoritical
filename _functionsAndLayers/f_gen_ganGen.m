% 20200911 by Dushan N. Wadduwage
% Generator network for GAN (same network in the example: https://www.mathworks.com/help/deeplearning/ug/train-generative-adversarial-network.html)

function dlnetGenerator = f_gen_ganGen(pram)
  
  % only work for the sizes 32 (with tconv4 stride 1) and 64 (with tconv4 stride 2)
  numFilters      = pram.numFilters;
  numLatentInputs = pram.Ncompressed_gen;

  filterSize = 5;
  projectionSize = [4 4 512];

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
      % transposedConv2dLayer(filterSize,1,'Stride',1,'Cropping','same','Name','tconv4')
      transposedConv2dLayer(filterSize,1,'Stride',2,'Cropping','same','Name','tconv4')
      tanhLayer('Name','tanh')];

  lgraphGenerator = layerGraph(layersGenerator);
  dlnetGenerator  = dlnetwork(lgraphGenerator);

end