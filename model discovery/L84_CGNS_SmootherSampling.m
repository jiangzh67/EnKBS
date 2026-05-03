function [x_mean, x_std, mu_s, x_s_std, x_samp] = L84_CGNS_SmootherSampling( ...
    dt, Nt, m_ens, sigma_x2, sigma_y2, sigma_z2, y_obs, z_obs, Theta)
%% l84 cgns filter/smoother + one conditional sample
% inputs: dt scalar, Nt scalar, m_ens scalar
%         sigma_x2, sigma_y2, sigma_z2 scalars
%         y_obs, z_obs: 1xNt or Ntx1
%         Theta: 36x1
% outputs: x_mean, x_std, mu_s, x_s_std, x_samp: 1xNt

    % sanity checks
    assert(numel(y_obs) == Nt && numel(z_obs) == Nt, 'y_obs and z_obs must have length Nt.');
    assert(numel(Theta) == 36, 'Theta must be 36x1.');

    y_obs = y_obs(:);
    z_obs = z_obs(:);

    sigma_x = sqrt(sigma_x2);

    invBB = diag([1/sigma_y2, 1/sigma_z2]);

    th = parseTheta36_L84(Theta);

    %% cgns filter
    mu_f = zeros(1, Nt);
    R_f  = zeros(1, Nt);

    mu_f(1) = 1.0;
    R_f(1)  = 0.1^2;


    for k = 1:Nt-1
        yk  = y_obs(k);   zk  = z_obs(k);
        ykp = y_obs(k+1); zkp = z_obs(k+1);

        dX = [ykp - yk; zkp - zk];

        phi = basis_phi_L84(yk, zk);

        a0 = th.x0.' * phi;
        a1 = th.x1.' * phi;

        A0 = [th.y0.' * phi; th.z0.' * phi];
        A1 = [th.y1.' * phi; th.z1.' * phi];

        innov = dX - (A0 + A1 * mu_f(k)) * dt;
        Kg    = R_f(k) * (A1.' * invBB);

        mu_f(k+1) = mu_f(k) + (a0 + a1 * mu_f(k)) * dt + Kg * innov;

        s   = A1.' * invBB * A1;
        dRf = (2*a1*R_f(k) + sigma_x2 - (R_f(k)^2) * s);
        R_f(k+1) = R_f(k) + dRf * dt;
    end

    x_mean = mu_f;
    x_std  = sqrt(R_f);

    %% cgns smoother
    mu_s = zeros(1, Nt);
    R_s  = zeros(1, Nt);

    mu_s(Nt) = mu_f(Nt);
    R_s(Nt)  = R_f(Nt);

    for k = Nt-1:-1:1
        yk1 = y_obs(k+1);  zk1 = z_obs(k+1);
        phi = basis_phi_L84(yk1, zk1);

        a0 = th.x0.' * phi;
        a1 = th.x1.' * phi;

        Rf1   = R_f(k+1);
        alpha = sigma_x2 / Rf1;

        dmu_rev = (-a0 - a1 * mu_s(k+1) + alpha * (mu_f(k+1) - mu_s(k+1)));
        mu_s(k) = mu_s(k+1) + dmu_rev * dt;

        dRs_rev = (-2*(a1 + alpha) * R_s(k+1) + sigma_x2);
        R_s(k)  = R_s(k+1) + dRs_rev * dt;
    end

    x_s_std = sqrt(R_s);

    %% cgns conditional sampling
    x_samp = zeros(1, Nt);

    x_samp(Nt) = mu_s(Nt) + sqrt(R_s(Nt)) * randn;

    for k = Nt-1:-1:1
        yk1 = y_obs(k+1);  zk1 = z_obs(k+1);
        phi = basis_phi_L84(yk1, zk1);

        a0 = th.x0.' * phi;
        a1 = th.x1.' * phi;

        Rf1   = R_f(k+1);
        alpha = sigma_x2 / Rf1;

        xnext = x_samp(k+1);
        drift_rev = (-a0 + alpha * mu_f(k+1) - (a1 + alpha) * xnext);

        x_samp(k) = xnext + drift_rev * dt + sigma_x * sqrt(dt) * randn;
    end

end

% helpers

function th = parseTheta36_L84(Theta)
    th = struct();
    th.x0 = Theta(1:6);
    th.x1 = Theta(7:12);
    th.y0 = Theta(13:18);
    th.y1 = Theta(19:24);
    th.z0 = Theta(25:30);
    th.z1 = Theta(31:36);
end

function phi = basis_phi_L84(y, z)
    phi = [1; y; z; y^2; z^2; y*z];
end
