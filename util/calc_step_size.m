function stepSize = calc_step_size(d,gx,step_max)    
    stepSize = min(step_max, d'*-gx/(d'*d));  
end