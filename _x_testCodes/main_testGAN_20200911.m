% 20200911 by Dushan N. Wadduwage
% test a deep autoencoder 

cd('../')
addpath(genpath('./_functionsAndLayers/'))
addpath('./_Datasets/')
addpath('./_ExtPatternsets/')

pram      = pram_init();
X         = f_get_dataset(pram);
trOptions = f_set_training_options_gan(pram);

Models.dlnetGenerator      = f_gen_ganGen(pram);
Models.dlnetDiscriminator  = f_gen_stdDisc(pram);
% analyzeNetwork(layerGraph(dlnetDiscriminator))

[nets info] = f_train_gan(X.Train,X.Val,Models,trOptions);
dlnet_gen   = nets.dlnetGenerator;  

XhatTest    = predict(dlnet_gen,dlarray(10*rand(1,1,pram.Ncompressed_gen,100),'SSCB'));
imagesc(imtile([XhatTest.extractdata]))


