
function dlnetEncoder = f_gen_linEnc(pram)
    
    Nx              = pram.Nx;
    Nc              = pram.Nc;
    Ncompressed     = pram.Ncompressed_gen;
    numFilters      = pram.numFilters;

    filterSize      = [4 4];
    N_middleLayers  = log2(Nx)-2;        
    N_filters       = numFilters;
    
    layersEncoder   = [
        imageInputLayer([Nx Nx Nc],'Normalization','none','Name','enc_in')
        convolution2dLayer(filterSize,N_filters,'Stride',2,'Padding',1,'Name','conv1')
        ];
    
    for i=2:N_middleLayers    
        N_filters       = 2*N_filters;
        layersEncoder   = [
            layersEncoder
            convolution2dLayer(filterSize,N_filters,'Stride',2,'Padding',1,'Name',sprintf('conv%d',i))
            batchNormalizationLayer('Name',sprintf('bn%d',i))            
            ];
    end
    
    layersEncoder   = [
        layersEncoder
        convolution2dLayer(filterSize,Ncompressed,'Name',sprintf('conv%d',i+1))
        ];
        
    lgraphEncoder   = layerGraph(layersEncoder);
    dlnetEncoder    = dlnetwork(lgraphEncoder);
    
end