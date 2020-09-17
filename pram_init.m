
function pram = pram_init()

  %% names
  pram.mic_typ      = 'DMD';                          % {'DMD','WGD'}
  pram.pattern_typ  = 'dmd_sim_rnd';                  % {'dmd_sim_rnd','dmd_exp','wgd_sim','wgd_exp'}
  pram.dataset      = 'andrewCells_dapi_20x_maxProj'; % {'minist','andrewCells_dapi_20x_maxProj','cells_h2ax'}
  pram.psf_typ      = 'gaussian';                     % {'gaussian',...}
  
  %% data size parameters
  pram.Nx      = 32;
  pram.Ny      = 32;
  pram.Nc      = 1;
  pram.Nt      = 16;

  %% compression parameters
  pram.compression_gen = 32;
  pram.compression_fwd = 4;
  
  pram.Ncompressed_gen = pram.Nx*pram.Ny/pram.compression_gen;            % dim non-linear feature space
  pram.Ncompressed_fwd = pram.Nx*pram.Ny/(pram.Nt*pram.compression_fwd);  % dim measurement space

  %% camera parameters
  pram.amp     = 1e8;                                % scaling factor from measured images to images in [0 1]
  pram.mu_rd   = 100;
  pram.sd_rd   = 10;
  pram.binR    = floor(sqrt(pram.compression_fwd*pram.Nt));
  
  %% network paramters
  pram.numFilters  = 64;
  pram.scale       = 0.01;                           % pramater of the leakyReLu

  %% training parameters
  pram.maxEpochs          = 200;
  pram.miniBatchSize      = 512;
  pram.initLearningRate   = 1;
  pram.learningRateFactor = .1;
  pram.dropPeriod         = round(pram.maxEpochs/4);
  pram.l2reg              = 0.0001;
  pram.excEnv             = 'auto';                 % {'auto','gpu','multi-gpu'}
end