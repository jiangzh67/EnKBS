function [Theta_long, sigma_y_hat, sigma_z_hat] = ...
    l84_estimate_theta_sigma(smask_long, x_path, y_obs, z_obs, dt, sigma_x, H, g)
% estimate theta and observation noises with constraint H*Theta = g
% inputs: smask_long is 33x1
%         x_path, y_obs, z_obs: Nt x 1 or 1 x Nt
%         dt, sigma_x: scalars
%         H is r x 36 and g is r x 1 or empty
% outputs: Theta_long: 36x1
%          sigma_y_hat, sigma_z_hat: scalars

    maxIter = 10;

    % ensure columns
    x = x_path(:);
    y = y_obs(:);
    z = z_obs(:);

    Nt = numel(x);
    assert(numel(y) == Nt && numel(z) == Nt, 'Length mismatch in x,y,z.');

    idx_k  = 1:(Nt-1);
    idx_kp = idx_k + 1;
    K = numel(idx_k);

    dx = x(idx_kp) - x(idx_k);
    dy = y(idx_kp) - y(idx_k);
    dz = z(idx_kp) - z(idx_k);

    xk = x(idx_k);
    yk = y(idx_k);
    zk = z(idx_k);

    % library order
    % 1, y, z, y^2, z^2, yz, x, xy, xz, xy^2, xz^2, xyz
    Phi = [ ...
        ones(K,1), ...
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
    ]; % K x 12

    smask_long = logical(smask_long(:));

    idx_nonconst = setdiff(1:36, [1 13 25], 'stable');
    assert(numel(idx_nonconst) == 33, 'Unexpected non-constant indexing length.');

    idx_const = [1 13 25];
    idx_keep_nonconst = idx_nonconst(smask_long);
    idx_act = [idx_const, idx_keep_nonconst];
    idx_act = idx_act(:);

    if nargin < 7 || isempty(H)
        H = zeros(0,36);
        g = zeros(0,1);
    end
    g = g(:);
    H_act = H(:, idx_act);

    if ~isempty(H_act)
        row_keep = any(abs(H_act) > 0, 2);
        H_act = H_act(row_keep, :);
        g_act = g(row_keep);
    else
        g_act = zeros(0,1);
    end

    % initialize sigma_y and sigma_z
    sigma_y_hat = sqrt( mean(dy.^2) / dt );
    sigma_z_hat = sqrt( mean(dz.^2) / dt );

    Theta_long = zeros(36,1);
    Theta_long_new = zeros(36,1);
    
    G  = Phi.' * Phi;
    bx = Phi.' * dx;
    by = Phi.' * dy;
    bz = Phi.' * dz;

    for it = 1:maxIter

        wx = 1/(sigma_x^2 * dt);
        wy = 1/(sigma_y_hat^2 * dt);
        wz = 1/(sigma_z_hat^2 * dt);

        Dx = wx * dt^2 * G;
        Dy = wy * dt^2 * G;
        Dz = wz * dt^2 * G;

        D_full = blkdiag(Dx, Dy, Dz);

        cx = wx * dt * bx;
        cy = wy * dt * by;
        cz = wz * dt * bz;

        c_full = [cx; cy; cz];

        D = D_full(idx_act, idx_act);
        c = c_full(idx_act);

        if isempty(H_act)
            Theta_act = D \ c;
        else
            Dinv_c  = D \ c;
            Dinv_Ht = D \ (H_act.');

            A   = H_act * Dinv_Ht;
            rhs = H_act * Dinv_c - g_act;
            lambda = A \ rhs;

            Theta_act = D \ (c - H_act.' * lambda);
        end
        
        Theta_long_new(:) = 0;
        Theta_long_new(idx_act) = Theta_act;
        
        Err = norm(Theta_long-Theta_long_new)/norm(Theta_long_new);
        if Err <= 10e-15
            break;
        end
        
        Theta_long = Theta_long_new;

        thx = Theta_long(1:12);
        thy = Theta_long(13:24);
        thz = Theta_long(25:36);

        pred_dx = dt * (Phi * thx);
        pred_dy = dt * (Phi * thy);
        pred_dz = dt * (Phi * thz);

        ry = dy - pred_dy;
        rz = dz - pred_dz;

        sigma_y_hat = sqrt( mean(ry.^2) / dt );
        sigma_z_hat = sqrt( mean(rz.^2) / dt );
    end
end
