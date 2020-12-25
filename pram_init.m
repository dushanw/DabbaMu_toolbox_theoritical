
function pram = pram_init()

  %% names
  pram.mic_typ      = 'DMD';                            % {'DMD','WGD'}
  pram.pattern_typ  = 'dmd_exp_tfm_mouse_20201224';     % {'dmd_exp_tfm_mouse_20201224',
                                                        %  'dmd_exp_tfm_mouse_20201222',
                                                        %  'dmd_exp_tfm_beads_7sls_20201219',
                                                        %  'dmd_exp_tfm_mouse20201219',  
                                                        %  'dmd_sim_rnd',
                                                        %  'dmd_exp',
                                                        %  'dmd_exp_tfm',
                                                        %  'dmd_exp_tfm_beads_3',
                                                        %  'dmd_exp_tfm_beads_4',
                                                        %  'dmd_exp_tfm_beads_8',
                                                        %  'wgd_sim',
                                                        %  'wgd_exp'
                                                        % }
  pram.dataset      = 'minist';                         % {'minist',
                                                        %  'andrewCells_fociW3_63x_maxProj',
                                                        %  'andrewCells_dapi_20x_maxProj',}
  pram.psf_typ      = 'gaussian';                       % {'gaussian',...}
  
  %% data size parameters
  pram.Nx      = 200;
  pram.Ny      = 200;
  pram.Nc      = 1;
  pram.Nt      = 250;

  %% compression parameters
  pram.compression_gen = 256;
  pram.compression_fwd = 1/pram.Nt;
  
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