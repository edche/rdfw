function [U,S,V, num_active, PX] = update_svd(U, S, V, M, PX, num_active, rows,cols)
    % Performs a rank one update to compute U*S*V = U_old*S_old*V_old + uv'
    % Reformulate this to [U , u]*M*[S, s]', need to provide appropriate M
     
    [Q_u, R_u] = qr(U(:,1:num_active),0); % was faster computing this than the rank-one QR update
    [Q_v, R_v] = qr(V(:,1:num_active),0);      

    % X = Q_u * S * Q_v'
    [U_s, S_s, V_s] = svd( R_u * M * R_v', 0);
    U(:,1:size(U_s,2)) = Q_u*U_s;
    U(:,size(U_s,2)+1:end) = 0;
    
    S(1:min(size(U_s,2), size(V_s,2))) = diag(S_s);
    S(min(size(U_s,2), size(V_s,2))+1:end) = 0;
    
    V(:,1:size(V_s,2)) = Q_v*V_s;       
    V(:,size(V_s,2)+1:end) = 0;
    active_atoms = sum(S(1:num_active) > S(1)*eps); 
    if active_atoms ~= num_active
        % Truncated the SVD, recalculate PX
        PX = PX - compute_PX(U(:,active_atoms+1:num_active),diag(S(active_atoms+1:num_active))*V(:,active_atoms+1:num_active)',rows,cols);        
        U(:,active_atoms+1:num_active) = 0;
        V(:,active_atoms+1:num_active) = 0;
        S(active_atoms+1:num_active) = 0;
    end
    
    num_active = active_atoms; 
    clear Q_u R_u Q_v R_v U_s S_S V_s;
end