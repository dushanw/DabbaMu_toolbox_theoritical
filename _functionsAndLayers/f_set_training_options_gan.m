

function trOptions = f_set_training_options_gan(pram)

  trOptions.numEpochs                   = pram.maxEpochs;
  trOptions.miniBatchSize               = pram.miniBatchSize;
  trOptions.learnRateGenerator          = 0.0002;
  trOptions.learnRateDiscriminator      = 0.0001;
  trOptions.gradientDecayFactor         = 0.5;
  trOptions.squaredGradientDecayFactor  = 0.999;
  trOptions.executionEnvironment        = pram.excEnv;
  trOptions.flipFactor                  = 0.1;
  
  trOptions.trailingAvgEncoder          = [];
  trOptions.trailingAvgSqEncoder        = [];
  trOptions.trailingAvgGenerator        = [];
  trOptions.trailingAvgSqGenerator      = [];
  trOptions.trailingAvgDiscriminator    = [];
  trOptions.trailingAvgSqDiscriminator  = [];
  
end