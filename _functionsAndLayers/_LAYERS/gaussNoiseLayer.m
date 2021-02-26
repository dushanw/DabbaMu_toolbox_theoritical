classdef gaussNoiseLayer < nnet.layer.Layer    
        
    properties
        sd
    end
        
    properties (Learnable)
    end
    
    methods
        function layer = gaussNoiseLayer(name,avgLaserAmp)
            layer.sd = single(sqrt(avgLaserAmp));                  
            layer.Name        = name;                  
        end

        function Z = predict(layer,X)
            noise = randn(size(X),'single')/layer.sd;
            Z = X+noise;            
        end
        
    end
end