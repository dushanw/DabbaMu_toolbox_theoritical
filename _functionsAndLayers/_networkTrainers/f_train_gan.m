% 20200911 by Dushan N. Wadduwage (wadduwage@fas.harvard.edu)
% custom training gan loop
% inputs:   XTrain, XVal, Models (as Models.dlnetGenerator & Models.dlnetDiscriminator), training options
% outputs:  nets (as nets.dlnetGenerator & nets.dlnetDiscriminator), training info 

% The latent space is Z (same as 'alpha' in the manuscript)
%   i.e. X_generated = Models.dlnetGenerator(Z)


function [nets info] = f_train_gan(XTrain,XVal,Models,opt)

  dlnetGenerator      = Models.dlnetGenerator;
  dlnetDiscriminator  = Models.dlnetDiscriminator;
  flipFactor          = opt.flipFactor;
  
  
  dlXVal              = dlarray(XVal  ,'SSCB');
  ZVal                = randn([dlnetGenerator.Layers(1).InputSize size(XVal,4)],'single');
  dlZVal              = dlarray(ZVal,'SSCB');
  
  if (opt.executionEnvironment == "auto" && canUseGPU) || opt.executionEnvironment == "gpu"
    dlXVal  = gpuArray(dlXVal);        
    dlZVal  = gpuArray(dlZVal);        
  end

  figure
  iteration       = 0;
  start           = tic;
  numTrainImages  = size(XTrain,4);

  for i = 1:opt.numEpochs
    XTrain = XTrain(:,:,:,randperm(numTrainImages));

    for ii=1:opt.miniBatchSize:numTrainImages-opt.miniBatchSize 
      iteration   = iteration + 1;
      
      X           = XTrain(:,:,:,ii:ii+opt.miniBatchSize-1);
      dlX         = dlarray(X, 'SSCB');
      Z           = randn([dlnetGenerator.Layers(1).InputSize size(X,4)],'single');
      dlZ         = dlarray(Z, 'SSCB');

      if (opt.executionEnvironment == "auto" && canUseGPU) || opt.executionEnvironment == "gpu"
          dlX = gpuArray(dlX);
          dlZ = gpuArray(dlZ);
      end

      % Evaluate the model gradients and the generator state
      [gradientsGenerator, gradientsDiscriminator, stateGenerator,losses_itr] = ...
        dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlZ, dlX, flipFactor, i, iteration);
      dlnetGenerator.State    = stateGenerator;
      allLosses(iteration,:)  = extractdata(losses_itr);

      % Update the discriminator network parameters.
      [dlnetDiscriminator.Learnables,opt.trailingAvgDiscriminator,opt.trailingAvgSqDiscriminator] = ...
        adamupdate(dlnetDiscriminator.Learnables, gradientsDiscriminator, ...
                   opt.trailingAvgDiscriminator , opt.trailingAvgSqDiscriminator, iteration, ...
                   opt.learnRateDiscriminator   , opt.gradientDecayFactor,        opt.squaredGradientDecayFactor);

      % Update the generator network parameters.
      [dlnetGenerator.Learnables,opt.trailingAvgGenerator,opt.trailingAvgSqGenerator] = ...
        adamupdate(dlnetGenerator.Learnables, gradientsGenerator, ...
                   opt.trailingAvgGenerator , opt.trailingAvgSqGenerator, iteration, ...
                   opt.learnRateGenerator   , opt.gradientDecayFactor   , opt.squaredGradientDecayFactor);

      % Every 20 iterations, display validation results                    
      if mod(iteration,20) == 0 || iteration == 1                        
        dlXGeneratedValidation  = predict(dlnetGenerator,dlZVal);

        I   = imtile(cat(2,extractdata(dlXVal),extractdata(dlXGeneratedValidation),zeros(size(dlXVal,1),10,1,size(dlXVal,4))));
        I   = rescale(I);
        imagesc(I);axis image

        % Update the title with training progress information.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title(...
          "Epoch: " + i + ", " + ...
          "Iteration: " + iteration + ", " + ...
          "Elapsed: " + string(D))

        drawnow
      end            
    end
  end

  nets.dlnetGenerator     = dlnetGenerator;
  nets.dlnetDiscriminator = dlnetDiscriminator;

  allLosses           = gather(allLosses);    
  info.lossGen        = allLosses(:,1);
  info.lossDisc       = allLosses(:,2);
end


% Model Gradients Function

function [gradientsGenerator, gradientsDiscriminator, stateGenerator, allLosses] = ...
  modelGradients(dlnetGenerator, dlnetDiscriminator, dlZ, dlX, flipFactor,epoch,iteration)

  % Calculate the predictions for real data with the discriminator network.
  dlYPred = forward(dlnetDiscriminator, dlX);

  % Calculate the predictions for generated data with the discriminator network.
  [dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
  dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated);

  % Convert the discriminator outputs to probabilities.
  probGenerated = sigmoid(dlYPredGenerated);
  probReal = sigmoid(dlYPred);

  % Calculate the score of the discriminator.
  scoreDiscriminator = ((mean(probReal)+mean(1-probGenerated))/2);

  % Calculate the score of the generator.
  scoreGenerator = mean(probGenerated);

  % Randomly flip a fraction of the labels of the real images.
  numObservations = size(probReal,4);
  idx = randperm(numObservations,floor(flipFactor * numObservations));

  % Flip the labels
  probReal(:,:,:,idx) = 1-probReal(:,:,:,idx);

  % Calculate the GAN loss.
  [lossGenerator, lossDiscriminator, allLosses] = ganLoss(probReal,probGenerated);

  % For each network, calculate the gradients with respect to the loss.
  gradientsGenerator = dlgradient(lossGenerator, dlnetGenerator.Learnables,'RetainData',true);
  gradientsDiscriminator = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);

end


function [lossGenerator, lossDiscriminator, allLosses] = ganLoss(probReal,probGenerated)

  % Calculate the loss for the discriminator network.
  lossDiscriminator =  -mean(log(probReal)) -mean(log(1-probGenerated));

  % Calculate the loss for the generator network.
  lossGenerator = -mean(log(probGenerated));

  allLosses = [lossGenerator lossDiscriminator];
end


