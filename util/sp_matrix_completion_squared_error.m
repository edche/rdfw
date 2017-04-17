function [f,dfx, gx] = sp_matrix_completion_squared_error(PX,dfx, rows,cols, vals)
    % Computes 
    % Sparse matrix completion squared error objective:
    %  min_X (1/2)(X(Y~=0) - Y(Y~=0))^2

    % Compute objective
    % X - Y only on (rows,cols)
    gx = PX - vals;

    f = gx'*gx/2;
    [m,n] = size(dfx);

    if nargout > 1
        % Compute gradient
        dfx = sparse(rows,cols,gx);
        if size(dfx,1) ~= m || size(dfx,2) ~= n
            dfx(m,n) = 0;
        end
    end
end
