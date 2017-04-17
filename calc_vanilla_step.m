function [U, S, V, num_active, step_type, step_size] = calc_vanilla_step(U, S, V, d_fw, gx, u,v,  num_active, delta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Algorithm 1: Calculates the regular Frank-Wolfe step.          %%%
    %%% -------------------------------------------------------------- %%%
    %%% Inputs:                                                        %%%    
    %%% U*S*V'                  Thin SVD of X                          %%%
    %%% d_fw                    Frank-Wolfe direction vectorized       %%%
    %%% gx                      Gradient vectorized                    %%%
    %%% u,v                     Leading SV pair of -grad(X)            %%%
    %%% num_active              Current Rank of X                      %%%
    %%% delta                   Nuclear norm bound                     %%% 
    %%% -------------------------------------------------------------- %%%
    %%% Outputs:                                                       %%%
    %%% U,S,V                   Updated thin SVD of X                  %%%
    %%% num_active              Updated rank of X                      %%%
    %%% step_type               Update flag to indicate regular FW     %%%
    %%% step_size               Optimal step-size for FW step          %%%
    %%% -------------------------------------------------------------- %%%
    %%% Written by Edward Cheung (eycheung@uwaterloo.ca) 2017          %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    global FRANK_WOLFE_STEP;
    step_type = FRANK_WOLFE_STEP;         
    step_size = calc_step_size(d_fw, gx, 1);
    S(1:num_active) = (1-step_size)*S(1:num_active);
    num_active = num_active + 1;
    S(num_active) = step_size*delta;
    U(:, num_active) = u;
    V(:, num_active) = v;
end
