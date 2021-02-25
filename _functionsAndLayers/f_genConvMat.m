% 20181019 by Dushan N. Wadduwage

function A_psf = f_genConvMat(psf,Ny,Nx)
    
    X_temp = zeros(Ny,Nx);
    for i=1:Ny*Nx 
        X_temp(:)=0;
        X_temp(i)=1;        
        Y_temp_psf = conv2(X_temp,psf,'same');
        A_psf(:,i)=Y_temp_psf(:);
    end
end

