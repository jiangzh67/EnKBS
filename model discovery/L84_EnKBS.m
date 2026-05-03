function [x_mean, x_std, x_s_mean, x_s_std,x_sampling] = L84_EnKBS(dt, Nt, m_ens, sigma_x2, sigma_y2, sigma_z2, y_obs, z_obs, Theta)
%% l84 enkbf + enkbs
% hidden state x
% observed path y_obs, z_obs
% theta = [theta_x0; theta_x1; theta_y0; theta_y1; theta_z0; theta_z1]
% inputs: dt scalar, Nt scalar, m_ens scalar
%         sigma_x2, sigma_y2, sigma_z2 scalars
%         y_obs, z_obs: 1xNt or Ntx1
%         Theta: 36x1
% outputs: x_mean, x_std, x_s_mean, x_s_std: 1xNt
%          x_sampling: 1x1xNt

    % sanity checks
    assert(numel(y_obs) == Nt && numel(z_obs) == Nt, 'y_obs and z_obs must have length Nt.');
    assert(numel(Theta) == 36, 'Theta must be 36x1.');
    
    sigma_x = sqrt(sigma_x2);
    sigma_y = sqrt(sigma_y2);
    sigma_z = sqrt(sigma_z2);

    Gamma_inv = diag([1/(sigma_y^2), 1/(sigma_z^2)]);

    th = parseTheta36_L84(Theta);

    
    x_ens = 1 + 0.1 * randn(1, m_ens);

    x_filt_ens = zeros(1, m_ens, Nt);
    x_filt_ens(:,:,1) = x_ens;

    x_mean = zeros(1, Nt);
    x_std  = zeros(1, Nt);
    x_mean(1) = mean(x_ens,2);
    x_std(1)  = std(x_ens,0,2);

    % forward noises reused in backward sampling
    dB_x = sqrt(dt) * randn(1, m_ens, Nt-1);
    
    dW_y = sqrt(dt) * randn(1, m_ens, Nt-1);
    dW_z = sqrt(dt) * randn(1, m_ens, Nt-1);

    %% enkbf
    for k = 2:Nt
        y_prev = y_obs(k-1);
        z_prev = z_obs(k-1);
        y_curr = y_obs(k);
        z_curr = z_obs(k);

        dY = [y_curr - y_prev; z_curr - z_prev];

        % hidden drift and observed drift
        f_vals = zeros(1, m_ens);
        h_vals = zeros(2, m_ens);

        phi = basis_phi_L84(y_prev, z_prev);
        a0x = th.x0.' * phi;   a1x = th.x1.' * phi;
        h0y = th.y0.' * phi;   h1y = th.y1.' * phi;
        h0z = th.z0.' * phi;   h1z = th.z1.' * phi;

        for e = 1:m_ens
            xe = x_ens(1,e);
            f_vals(1,e) = a0x + a1x * xe;
            h_vals(1,e) = h0y + h1y * xe;
            h_vals(2,e) = h0z + h1z * xe;
        end

        x_bar = mean(x_ens, 2);
        h_bar = mean(h_vals, 2);

        X_anom = x_ens - x_bar;
        H_anom = h_vals - h_bar;
        P_xh = (X_anom * H_anom.') / (m_ens - 1);

        K = P_xh * Gamma_inv;

        % forecast step
        x_pred = x_ens + f_vals * dt + sigma_x * dB_x(:,:,k-1);

       
        for e = 1:m_ens
            obs_pert = [sigma_y * dW_y(1,e,k-1); ...
                        sigma_z * dW_z(1,e,k-1)];
            innov = dY - h_vals(:,e) * dt + obs_pert; % 2x1
            x_ens(1,e) = x_pred(1,e) + K * innov;
        end

        x_filt_ens(:,:,k) = x_ens;
        x_mean(k) = mean(x_ens,2);
        x_std(k)  = std(x_ens,0,2);
    end

    %% enkbs

    x_s_ens = zeros(1, m_ens, Nt);
    x_s_ens(:,:,Nt) = x_filt_ens(:,:,Nt);

    x_s_mean = zeros(1, Nt);
    x_s_std  = zeros(1, Nt);
    x_s_mean(Nt) = mean(x_s_ens(:,:,Nt),2);
    x_s_std(Nt)  = std(x_s_ens(:,:,Nt),0,2);

    for k = Nt-1:-1:1
        Xf = x_filt_ens(:,:,k+1);
        Pf = var(Xf, 0, 2);

        y_next = y_obs(k+1);
        z_next = z_obs(k+1);
        phi = basis_phi_L84(y_next, z_next);

        a0x = th.x0.' * phi;
        a1x = th.x1.' * phi;

        for e = 1:m_ens
            xs = x_s_ens(1,e,k+1);
            xf = x_filt_ens(1,e,k+1);

            f_s = a0x + a1x * xs;

            dB = dB_x(1,e,k);

            % backward attraction
            attract = sigma_x2 * (xs - xf) / Pf;

            x_s_ens(1,e,k) = xs - f_s*dt - sigma_x*dB - attract*dt;
        end

        x_s_mean(k) = mean(x_s_ens(:,:,k),2);
        x_s_std(k)  = std(x_s_ens(:,:,k),0,2);
    end
    
    x_sampling = x_s_ens(1,1,:);
    
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

