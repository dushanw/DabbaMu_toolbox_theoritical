
function lgraphAutoEnc = f_gen_linAutoEnc(pram)

  Nx          = pram.Nx;
  Ny          = pram.Ny;                  % for now Nx = Ny is a must
  Nc          = pram.Nc;
  Ncompressed = pram.Ncompressed_gen;
  numFilters  = pram.numFilters;
  scale       = pram.scale;
  
  filterSize      = [4 4];
  N_middleLayers  = log2(Nx)-2;           % for now Nx = Ny is a must
  N_filters       = numFilters;        
  
  %% Enc
  switch pram.encType
    case 'conv'  
      layersEncoder   = [
          imageInputLayer([Ny Nx Nc],'Normalization','none','Name','enc_in')
          convolution2dLayer(filterSize,N_filters,'Stride',2,'Padding',1,'Name','enc_conv1')
          % leakyReluLayer(scale,'Name','enc_lrelu1')
          ];    
      for i=2:N_middleLayers    
          N_filters       = 2*N_filters;
          layersEncoder   = [
              layersEncoder
              convolution2dLayer(filterSize,N_filters,'Stride',2,'Padding',1,'Name',sprintf('enc_conv%d',i))
              batchNormalizationLayer('Name',sprintf('enc_bn%d',i))
              % leakyReluLayer(scale,'Name',sprintf('enc_lrelu%d',i))
              ];
      end    
      layersEncoder   = [
          layersEncoder
          convolution2dLayer(filterSize,Ncompressed,'Name',sprintf('enc_conv%d',i+1))
          ];
    case 'fc_rnd'
      layersEncoder   = [
          imageInputLayer([Ny Nx Nc],'Normalization','none','Name','enc_in')
          fullyConnectedLayer(Ncompressed,'Name','enc_fc',...
                              'Weights',subf_wi_rand([Ncompressed, Ny*Nx*Nc]),...
                              'Bias',zeros(Ncompressed,1),...                              
                              'BiasLearnRateFactor',0)
          ];        
    case 'fc_rnd_fixed'
      layersEncoder   = [
          imageInputLayer([Ny Nx Nc],'Normalization','none','Name','enc_in')
          fullyConnectedLayer(Ncompressed,'Name','enc_fc',...
                              'Weights',subf_wi_rand([Ncompressed, Ny*Nx*Nc]),...
                              'Bias',zeros(Ncompressed,1),...
                              'WeightLearnRateFactor',0,...
                              'BiasLearnRateFactor',0)
          ];                
    case 'fc_had'            
      layersEncoder   = [
          imageInputLayer([Ny Nx Nc],'Normalization','none','Name','enc_in')
          fullyConnectedLayer(Ncompressed,'Name','enc_fc',...
                              'Weights',subf_wi_had([Ncompressed, Ny*Nx*Nc]),...
                              'Bias',zeros(Ncompressed,1),...                              
                              'BiasLearnRateFactor',0)
          ];        
    case 'fc_had_fixed'
      layersEncoder   = [
          imageInputLayer([Ny Nx Nc],'Normalization','none','Name','enc_in')
          fullyConnectedLayer(Ncompressed,'Name','enc_fc',...
                              'Weights',subf_wi_had([Ncompressed, Ny*Nx*Nc]),...
                              'Bias',zeros(Ncompressed,1),...
                              'WeightLearnRateFactor',0,...
                              'BiasLearnRateFactor',0)
          ];   
    case 'fc_rnd_posNormWeights'
      layersEncoder   = [
          imageInputLayer([Ny Nx Nc],'Normalization','none','Name','enc_in')
          projectAndReshape_posNormWeights_Layer([1 1 Ncompressed], Nc,'enc_posNormFc', subf_wi_rand([Ncompressed, Ny*Nx*Nc]))
          ];                
    case 'fc_rnd_posNormWeights_fixed'
      layersEncoder   = [
          imageInputLayer([Ny Nx Nc],'Normalization','none','Name','enc_in')
          projectAndReshape_posNormWeights_Layer_fixed([1 1 Ncompressed], Nc,'enc_posNormFc', subf_wi_rand([Ncompressed, Ny*Nx*Nc]))
          ];                
      
  end

  %% Gen        
  layersGenerator = [        
      transposedConv2dLayer(filterSize,2.^(N_middleLayers-1)*numFilters,'Name','gen_tconv1')
      batchNormalizationLayer('Name','gen_bn1')
      reluLayer('Name','gen_relu1')];
  for i=2:N_middleLayers    
      N_filters   = 2.^(N_middleLayers-i)*numFilters;
      layersGenerator = [
          layersGenerator 
          transposedConv2dLayer(filterSize,N_filters,'Stride',2,'Cropping',1,'Name',sprintf('gen_tconv%d',i))
          batchNormalizationLayer('Name',sprintf('gen_bn%d',i))
          reluLayer('Name',sprintf('gen_relu%d',i))];
  end    
  layersGenerator = [
          layersGenerator 
          transposedConv2dLayer(filterSize,Nc,'Stride',2,'Cropping',1,'Name',sprintf('gen_tconv%d',i+1))
          tanhLayer('Name','gen_tanh')];

  %% auto encoder
  switch pram.encNoise
    case 'none'
      layersAutoEnc   = [
              layersEncoder              
              layersGenerator
              regressionLayer('Name','out')
                        ];      
    case 'gauss'    
      layersAutoEnc   = [
              layersEncoder
              gaussNoiseLayer('noise',sqrt(pram.avgCounts))
              layersGenerator
              regressionLayer('Name','out')
                        ];
    case 'poiss'
      layersAutoEnc   = [
              layersEncoder
              aproxPoissonLayer('noise',sqrt(pram.avgCounts))
              layersGenerator
              regressionLayer('Name','out')
                        ];
  end
      

  lgraphAutoEnc   = layerGraph(layersAutoEnc);   
end

function weights = subf_wi_rand(sz)  
  rng(1);
  weights = rand(sz);
end

function weights = subf_wi_had(sz)  
  rng(1);
  H       = hadamard(sz(2));
  idx     = randperm(sz(2));
  weights = H(idx(1:sz(1)),:);  
end











