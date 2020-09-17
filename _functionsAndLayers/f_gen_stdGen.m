
function dlnetGenerator = f_gen_stdGen(pram)
    
    Nx              = pram.Nx;
    Nc              = pram.Nc;
    Ncompressed     = pram.Ncompressed_gen;
    numFilters      = pram.numFilters;
    
    projectionSize  = [4 4 512];
    filterSize      = 4;
    N_middleLayers  = log2(Nx)-2;    
    
    layersGenerator = [
        imageInputLayer([1 1 Ncompressed],'Normalization','none','Name','in');
        projectAndReshapeLayer(projectionSize,Ncompressed,'proj');
        % transposedConv2dLayer(filterSize,2.^(N_middleLayers-1)*numFilters,'Name','tconv1')
        % batchNormalizationLayer('Name','bn1')
        reluLayer('Name','relu1')];

    for i=2:N_middleLayers    
        N_filters   = 2.^(N_middleLayers-i)*numFilters;
        layersGenerator = [
            layersGenerator 
            transposedConv2dLayer(filterSize,N_filters,'Stride',2,'Cropping',1,'Name',sprintf('tconv%d',i))
            batchNormalizationLayer('Name',sprintf('bn%d',i))
            reluLayer('Name',sprintf('relu%d',i))];
    end
    
    layersGenerator = [
            layersGenerator 
            transposedConv2dLayer(filterSize,Nc,'Stride',2,'Cropping',1,'Name',sprintf('tconv%d',i+1))
            tanhLayer('Name','tanh')];
        

    lgraphGenerator = layerGraph(layersGenerator);
    dlnetGenerator = dlnetwork(lgraphGenerator);
end