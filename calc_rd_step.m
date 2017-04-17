function [U, S, V,  num_active, step_type, step_size, d, B_k, PX] = calc_rd_step(U, S, V, PX, gx, dfx, num_active, delta, rows, cols, B_k, step_type, OPTIONS)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Algorithm 3: Rank-Drop Frank-Wolfe                             %%%
    %%% -------------------------------------------------------------- %%%
    %%% Inputs:                                                        %%%
    %%% U*S*V'                  Thin SVD of X                          %%%
    %%% PX                      X vectorized onto observed set         %%%
    %%% [gx, dfx]               Gradient [vector, sparse matrix]       %%%
    %%% num_active              Current Rank of X                      %%%
    %%% delta                   Nuclear norm bound                     %%%
    %%% rows, cols              row/col indices of observed set        %%%
    %%% B_k                     Bound on objective value               %%%
    %%% step_type               Flag indicating last step used         %%%
    %%% OPTIONS                 SVD calc options (max iter, tol, etc)  %%%
    %%% -------------------------------------------------------------- %%%
    %%% Outputs:                                                       %%%
    %%% U,S,V                   Updated thin SVD of X                  %%%
    %%% num_active              Updated rank of X                      %%%
    %%% step_type               Update flag to indicate regular FW     %%%
    %%% step_size               Optimal step-size for FW step          %%%
    %%% B_k                     Updated bound on objective value       %%%
    %%% PX                      Updated projection of X                %%%
    %%% -------------------------------------------------------------- %%%
    %%% Written by Edward Cheung (eycheung@uwaterloo.ca) 2017          %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    global RANK_DROP_STEP;
    fx = 0.5*(gx'*gx);    
    if step_type == RANK_DROP_STEP
        [u,~,v] = svds(-dfx, 1, 'largest', OPTIONS);
        d = delta*compute_PX(u,v',rows,cols) - PX;
        B_k = calc_opt_gap(d, fx, gx, B_k); 
        [U, S, V, num_active, step_type, step_size] = calc_vanilla_step(U, S, V, d, gx, u,v,  num_active, delta);       
    else
        [rd_u, rd_v, step_size] = calc_rank_drop_dir(U, S, V, dfx, num_active, delta);
        d_rd = PX - delta*compute_PX(rd_u, rd_v', rows, cols);
        rd_fx = sp_matrix_completion_squared_error(PX + step_size*d_rd, dfx, rows, cols, PX - gx);
        if rd_fx <= fx 
            S(1:num_active) = (1+step_size)*S(1:num_active);
            num_active = num_active + 1;
            S(num_active) = -step_size*delta;
            U(:, num_active) = rd_u;
            V(:, num_active) = rd_v;
            d = d_rd;
            step_type = RANK_DROP_STEP;
        else      
            [u,~,v] = svds(-dfx, 1, 'largest', OPTIONS);
            d = delta*compute_PX(u,v',rows,cols) - PX;            
            [U, S, V, num_active, step_type, step_size] = calc_vanilla_step(U, S, V, d, gx, u,v,  num_active, delta);            
            B_k = calc_opt_gap(d, fx, gx, B_k);                 
        end               
    end
end