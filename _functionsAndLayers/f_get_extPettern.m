
function E = f_get_extPettern(pram)

  switch pram.pattern_typ
    case 'rnd'
      E = rand([pram.Ny pram.Nx pram.Nt])-.5; % for DMDs
  end

end