function [u,v, step_size] = calc_rank_drop_dir(U, S, V, dfx, num_active, delta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Algorithm 2: Compute Rank-Drop Direction                       %%%
    %%% -------------------------------------------------------------- %%%
    %%% Inputs:                                                        %%%
    %%% U*S*V'                  Thin SVD of X                          %%%
    %%% d_fw                    Frank-Wolfe direction vectorized       %%%
    %%% num_active              Current Rank of X                      %%%
    %%% delta                   Nuclear norm bound                     %%%
    %%% -------------------------------------------------------------- %%%
    %%% Outputs:                                                       %%%
    %%% u*v'                    Optimal rank-drop matrix               %%%
    %%% step_size               Optimal step-size                      %%%
    %%% -------------------------------------------------------------- %%%
    %%% Written by Edward Cheung (eycheung@uwaterloo.ca) 2017          %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    kappa = (delta - sum(S(1:num_active)))/2;
    W = U(:,1:num_active)'*dfx*V(:,1:num_active);
    S_inv = diag(1./S(1:num_active));

    if kappa >= S(num_active)
        % Interior Case        
        lam_arr = eig(-diag(S(1:num_active))*W);
        best_val = -Inf;
        
        % Rank-Drop Candidate
        s = zeros(num_active,1);
        t = zeros(num_active,1);       
    
        % Case 1 of KKT, norm(t) < 1, s'*t > 0
        for lam_ind = 1:length(lam_arr)           
           if isreal(lam_arr(lam_ind))
               lambda = lam_arr(lam_ind); 
               [s_cand, t_cand, sig_cand, val] = solve_rd_subproblem(W, S_inv, lambda, kappa);               
               if val > best_val && sig_cand <= 1
                   s = s_cand;
                   t = t_cand;
                   best_val = val;
               end
           end
        end
        
        if norm(s) ~= 0          
            alpha = 1/(s'*S_inv*t);
            step_size = alpha/(delta - alpha);            
        else
            % Annulus Case
            [s,~] = eigs(0.5*(W + W'), S_inv, 1, 'LA');
            s = s/norm(s);
            t = s;
            step_size = 1/(delta*s'*S_inv*s - 1);
        end              
    else
        % Annulus Case
        [s,~] = eigs(0.5*(W + W'), S_inv, 1, 'LA');
        s = s/norm(s);
        t = s;
        step_size = 1/(delta*s'*S_inv*s - 1);
    end
    u = U(:,1:num_active)*s;
    v = V(:,1:num_active)*t;
end


function [s, t, sig, val] = solve_rd_subproblem(W, S_inv, lambda, kappa)
    M = -0.5*(W + lambda*S_inv);            
    % Solve it via Inverse Iteration    
    [s,t] = inverse_iter(M);        
    sig = 1/(kappa*t'*S_inv*s);
    if sig < 0
        sig = -sig;
        t = -t;
    end
    val = sig*s'*W*t;
end       

function [s,t] = inverse_iter(M)
    % inverse iteration to compute smallest eigenvalue
    r = size(M,1);    
    s = rand(r,1);
    t = rand(r,1);

    M_inv = inv(M);
    for i = 1:10
        t = M_inv*t;
        s = M_inv'*s;
        t = t/norm(t);
        s = s/norm(s);
    end
end