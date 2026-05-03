function [smask_long, CE_long, S_mask, CE_vals] = ...
    l84_causation_entropy_mask(x_hat, y_obs, z_obs, thrCE)
% causation-entropy mask for the non-constant library terms
% inputs: x_hat, y_obs, z_obs: Nt x 1 or 1 x Nt
%         thrCE: scalar
% outputs: smask_long, CE_long: 33x1
%          S_mask, CE_vals: 3x11

    % column vectors
    x_hat = x_hat(:);
    y_obs = y_obs(:);
    z_obs = z_obs(:);

    Nt = numel(x_hat);
    assert(numel(y_obs)==Nt && numel(z_obs)==Nt, 'Input length mismatch.');

    idx_t  = 1:1:(Nt-1);
    idx_tp = idx_t + 1;
    K = numel(idx_t);

    % non-constant library order
    % y, z, y^2, z^2, yz, x, xy, xz, xy^2, xz^2, xyz
    xk = x_hat(idx_t);
    yk = y_obs(idx_t);
    zk = z_obs(idx_t);

    Fmat = [ ...
        yk, ...
        zk, ...
        yk.^2, ...
        zk.^2, ...
        yk.*zk, ...
        xk, ...
        xk.*yk, ...
        xk.*zk, ...
        xk.*(yk.^2), ...
        xk.*(zk.^2), ...
        xk.*yk.*zk ...
    ];

    M = size(Fmat,2);
    
    R_feat = cov(Fmat);
    logdet_Rfeat = log(det(R_feat));

    CE_vals = zeros(3, M);
    S_mask  = false(3, M);

    X_targets = [ x_hat(idx_tp), y_obs(idx_tp), z_obs(idx_tp) ]; % K x 3

    for eq = 1:3
        X = X_targets(:,eq);

        V = [X, Fmat];
        R_all = cov(V);
        logdet_Rall = log(det(R_all));

        for m = 1:M
            keep = true(1,M);
            keep(m) = false;

            R_Y = R_feat(keep, keep);
            ldY = log(det(R_Y));

            cols_XY = [1, 1 + find(keep)];
            R_XY = R_all(cols_XY, cols_XY);
            ldXY = log(det(R_XY));

            CE = 0.5*( ldXY - ldY - logdet_Rall + logdet_Rfeat );

            CE_vals(eq,m) = CE;
            S_mask(eq,m)  = (CE >= thrCE);
        end
    end

    smask_long = reshape(S_mask.', [], 1);
    CE_long    = reshape(CE_vals.', [], 1);

end
