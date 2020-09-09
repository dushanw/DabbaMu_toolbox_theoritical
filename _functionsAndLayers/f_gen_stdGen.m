
function dlnetGenerator = f_gen_stdGen(Nx,Nc,Ncompressed,numFilters)
    
    filterSize      = [4 4];
    N_middleLayers  = log2(Nx)-2;    
    
    layersGenerator = [
        imageInputLayer([1 1 Ncompressed],'Normalization','none','Name','in');
        transposedConv2dLayer(filterSize,2.^(N_middleLayers-1)*numFilters,'Name','tconv1')
        batchNormalizationLayer('Name','bn1')
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