classdef aproxPoissonLayer < nnet.layer.Layer    
        
    properties
        avgLaserAmp
    end
        
    properties (Learnable)
    end
    
    methods
        function layer = aproxPoissonLayer(name,avgLaserAmp)
            layer.avgLaserAmp = avgLaserAmp;                  
            layer.Name        = name;                  
        end

        function Z = predict(layer,X)
            % X = abs(X)*layer.avgLaserAmp;
            X = (X-min(X(:)))*layer.avgLaserAmp+1;
            noise_gauss = rand(size(X));
            Z = sqrt(X) .* noise_gauss + X;
            Z = Z/layer.avgLaserAmp;
            
%            Z = X;
        end
        
    end
end