function [B_k, opt_gap] = calc_opt_gap(d_fw, fx, gx, B)   
    % Updates the optimality gap and the bound on the objective value
    B_k = min(fx, max(B, fx + d_fw'*gx));
    opt_gap = fx - B_k;
end