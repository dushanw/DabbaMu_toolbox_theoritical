
function dlnet_gen = f_get_gen(pram,net_autoEnc)

  lgraph_gen  = layerGraph(net_autoEnc);
  layerNames  = {lgraph_gen.Layers(:).Name};
  lgraph_gen  = removeLayers(lgraph_gen,layerNames(contains(layerNames,'enc')));
  lgraph_gen  = addLayers(lgraph_gen,imageInputLayer([1 1 pram.Ncompressed_gen],'Normalization','none','Name','gen_in'));
  lgraph_gen  = connectLayers(lgraph_gen,'gen_in','gen_tconv1');
  lgraph_gen  = removeLayers(lgraph_gen,'out');

  dlnet_gen   = dlnetwork(lgraph_gen);

end