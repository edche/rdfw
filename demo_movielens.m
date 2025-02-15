%% Demo Movielens
% Original Data can be found here: https://grouplens.org/datasets/movielens/
% Code is currently optimized for sparse gradient.
% If gradient is dense, code can still be used but may be suboptimal.

% If you wish to use this code for applications other than Matrix
% Completion, update the sp_matrix_completion_squared_error.m file to
% reflect the objective and gradient calculations.
% calc_step_size.m may also need to be updated based upon the optimal step
% length for a new objective.

clc;
clear all;
close all;
warning off;

addpath('mex');
addpath('util');

%%%%%%% Load MovieLens dataset %%%%%%%%%%
data_size = '100k'; % 100k, 1m, 10m, 20m
addpath(sprintf('data/ml-%s/', data_size));
load('Y.mat');
[m,n] = size(Y);

%%%%%%% PARAMS %%%%%%%
orig_delta = 3; % will get scaled by Frobenius norm
T = 1000; % Max Iterations
max_rank = 300; % Estimate on maximum rank to allow for preallocation
tol = 10^-2;
num_trials = 2;
tra_size = 0.5;
tst_size = 0.25;
save_files = 1;
display_interval = 25; % Determine how often you want to see updates

%%%% SVD solver options %%%%%%%
OPTIONS.tol = 1E-10; %       Convergence tolerance                     
OPTIONS.maxit = 50; %    Maximum number of iterations.                

%%%%%% Centre Data %%%%%%%%
nrm_Y = norm(Y,'fro');
[orig_rows, orig_cols, orig_vals] = find(Y);
orig_vals = orig_vals - mean(orig_vals);
orig_vals = orig_vals/std(orig_vals);

%%%%% Pre-allocate Statistics arrays %%%%%

rank_track = zeros(T+1, num_trials);
error_track = zeros(T+1, num_trials);
rel_opt_gap = zeros(T+1, num_trials);
iter_times = zeros(T+1,num_trials);
t_track = zeros(num_trials,1);
final_ranks = zeros(num_trials,1);
final_ranks_max = zeros(num_trials,1);
RMSE_vld = zeros(num_trials,1);
RMSE_tst = zeros(num_trials,1);

%%% Global Variables to indicate Frank-Wolfe Step Flags
global FRANK_WOLFE_STEP;
global RANK_DROP_STEP;
   
FRANK_WOLFE_STEP = 1;
RANK_DROP_STEP = 0;
fprintf('Staring Movielens %s\n', data_size);   
disp('################################')

seed = 1234567;
randn('state',seed);
rand('state',seed);

for trial_id = 1:num_trials

    fprintf('Trial #%d\n', trial_id);
    disp('################################')
    
    %%%%% Make test set %%%%
    idx = randperm(length(orig_vals));

    tra_idx = idx(1:floor(length(orig_vals)*tra_size));
    vld_idx = idx(ceil(length(orig_vals)*tra_size):floor(length(orig_vals)*(1-tst_size)));
    tst_idx = idx(ceil(length(orig_vals)*(1-tst_size)):end);
    clear idx;
    tra_data = sparse(orig_rows(tra_idx), orig_cols(tra_idx), orig_vals(tra_idx));    
    tra_data(m,n) = 0; % Ensure that the matrix is appropriate dimension

    rows = orig_rows(tra_idx);
    cols = orig_cols(tra_idx);
    vals = orig_vals(tra_idx);          

    vld_rows = orig_rows(vld_idx);
    vld_cols = orig_cols(vld_idx);
    vld_vals = orig_vals(vld_idx);

    tst_rows = orig_rows(tst_idx);
    tst_cols = orig_cols(tst_idx);
    tst_vals = orig_vals(tst_idx);

    num_ratings = length(tra_idx);
    delta = orig_delta*norm(tra_data,'fro');

    %% Preallocate Frank-Wolfe Arrays
    gx = zeros(num_ratings,1); % vec(gradient) on observed set
    d = zeros(num_ratings,1); % vec(final direction) on observed set
    PX = zeros(num_ratings,1); % vec(X) on observed set     
    dfx = sparse(m,n); % sparse gradient
    d_fw = zeros(num_ratings,1); % vec(Frank-Wolfe step) on observed set

    % Keep track of SVD as we go
    U = zeros(m, max_rank); 
    V = zeros(n, max_rank);
    S = zeros(max_rank,1);
    
    num_active = 1;
    B_k = 0;

    for t = 0:T-1            
        iter_start = tic;
       
        [fx, dfx, gx] = sp_matrix_completion_squared_error(PX,dfx,rows,cols,vals);
        
        opt_gap = fx - B_k;        

        if t == 0
            % For the first iteration, perform regular Frank-Wolfe step
            [u,~,v] = svds(-dfx, 1, 'largest', OPTIONS);
            U(:,1) = u;
            V(:,1) = v;
            step_type = 'vanilla';
            d = delta*compute_PX(u, v', rows, cols);
            step_size = calc_step_size(d, gx, 1);
            S(1) = step_size*delta;
            PX = step_size*d;                
            [B_k, opt_gap] = calc_opt_gap(d, fx, gx, B_k);
            rank_track(t+1, trial_id) = num_active;
        else
            % Perform Rank-Drop step            
            [U, S, V,  num_active, step_type, step_size, d, B_k, PX] = ...
                calc_rd_step(U, S, V, PX, gx, dfx, num_active, delta, rows, cols, B_k, step_type, OPTIONS);
            PX = PX + step_size*d;
            M = diag(S(1:num_active));          
            [U,S,V, num_active, PX] = update_svd(U, S, V, M, PX, num_active, rows, cols);
            rank_track(t+1, trial_id) = sum(S > 1E-6);                                  

            % Check to see if the solution is numerically accurate/feasible
            diff = norm(PX - compute_PX(U(:,1:num_active), diag(S(1:num_active))*V(:,1:num_active)',rows, cols));
            if diff > 1E-5
                disp('Getting a bit numerically bad, recalculate PX');
                PX = compute_PX(U(:,1:num_active), diag(S(1:num_active))*V(:,1:num_active)',rows, cols);
            end

            if sum(S(1:num_active)) > (1+ 1E-6)*delta
               input('Infeasible solution'); 
            end
        end

        error_track(t+1, trial_id) = fx;
        rel_opt_gap(t+1, trial_id) = abs(opt_gap/(B_k));
        num_active = sum(S(1:num_active) > 0);                                      

        % Check if solution has converged
        if abs(rel_opt_gap(t+1, trial_id))  < tol
            fprintf('Converged after %d iterations.\n', t+1);
            error_track(t+1, trial_id) = fx;
            rel_opt_gap(t+1, trial_id) = abs(opt_gap/(B_k));
            iter_times(t+1, trial_id) =  toc(iter_start);          
            rank_track(t+1, trial_id) = num_active;
            predict = compute_PX(U(:,1:num_active), diag(S(1:num_active))*V(:,1:num_active)', vld_rows, vld_cols);
            RMSE = sqrt(sumsqr(predict - vld_vals)/length(vld_vals));   
            break;
        end


        % Update iteration statistics
        iter_times(t+1, trial_id) =  toc(iter_start);            
        predict = compute_PX(U(:,1:num_active), diag(S(1:num_active))*V(:,1:num_active)', vld_rows, vld_cols);
        RMSE = sqrt(sumsqr(predict - vld_vals)/length(vld_vals));
        predict = compute_PX(U(:,1:num_active), diag(S(1:num_active))*V(:,1:num_active)', tst_rows, tst_cols);        
        tst_RMSE = sqrt(sumsqr(predict - tst_vals)/length(tst_vals));

        if mod(t+1,display_interval) == 0
            fprintf('Iteration: %d Error: %f  Rel_opt_gap: %f  Rank: %d, Time: %f  RMSE = %f  Test RMSE = %f\n', ...
                t+1,...                    
                error_track(t+1, trial_id),...
                rel_opt_gap(t+1, trial_id),...
                rank_track(t+1, trial_id),...
                sum(iter_times(1:t+1, trial_id)),...
                RMSE, ...
                tst_RMSE);
        end
    end
    
    RMSE_vld(trial_id) = RMSE;
    predict = compute_PX(U(:,1:num_active), diag(S(1:num_active))*V(:,1:num_active)', tst_rows, tst_cols);        
    RMSE_tst(trial_id) = sqrt(sumsqr(predict - tst_vals)/length(tst_vals));    
    t_track(trial_id) = t+1;
    final_ranks(trial_id) = rank_track(t+1, trial_id);
    final_ranks_max(trial_id) = max(rank_track(1:t+1, trial_id));
    
    fprintf('-----------------------\n');
    fprintf('Finished\n');
    fprintf('Relative opt gap = %f\n', rel_opt_gap(t+1, trial_id));
    fprintf('Final Rank = %d\n', rank_track(t+1, trial_id));
    fprintf('Max Rank = %d\n', max(rank_track(1:t+1, trial_id)));
    fprintf('Iteration Time = %f\n', sum(iter_times(:, trial_id)));
    fprintf('Validation RMSE = %f\n', RMSE);  
    fprintf('Test RMSE = %f\n', RMSE_tst(trial_id))
    fprintf('-----------------------\n');
end


if save_files
    save(sprintf('results/%s-ml%s.mat', 'demo', data_size), ...
            'error_track', 'rank_track', 'rel_opt_gap', 'iter_times', 't_track', ...
            'final_ranks', 'final_ranks_max', 'RMSE_tst', 'RMSE_vld', 'num_trials');
end

