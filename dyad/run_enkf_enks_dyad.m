function [rms_f_enkbf, rms_s_enkbs] = run_enkf_enks_dyad( ...
    m_ens, dt, N, ...
    u_truth, v_truth, ...
    d_u, d_v, c, F_u, F_v, sigma_u, sigma_v)
% run dyad enkbf/enkbs for one ensemble size
% inputs: m_ens, dt, N: scalars
%         u_truth, v_truth: 1xN
%         remaining inputs are model and noise parameters
% outputs: rms_f_enkbf and rms_s_enkbs: scalars
    
    %% enkbf

    % initial hidden ensemble
    v_ens = v_truth(1) + randn(m_ens,1);

    v_filt_ens  = zeros(m_ens,N);
    v_filt_ens(:,1) = v_ens;

    dW_v_ens = sqrt(dt) * randn(m_ens,N-1);

    dW_u_obs = sqrt(dt) * randn(m_ens,N-1);
   
    v_f_mean = zeros(1,N);
    v_f_mean(1) = mean(v_ens);

    for i = 2:N
        u_prev = u_truth(i-1);
        u_curr = u_truth(i);
        dY     = u_curr - u_prev;

        v_bar = mean(v_ens);

        % hidden drift and observed drift
        f_vals = -d_v * v_ens - c * u_prev^2 + F_v;
        h_vals = (-d_u + c * v_ens) * u_prev + F_u;

        dv = v_ens - v_bar;
        dh = h_vals - mean(h_vals);
        P_vh = (dv' * dh) / (m_ens - 1);

        K = P_vh / (sigma_u^2);

        dB_v = dW_v_ens(:,i-1);

        % forecast step
        v_pred = v_ens + f_vals*dt + sigma_v * dB_v;

        innovation = dY - h_vals*dt + sigma_u * dW_u_obs(:,i-1);

        v_ens = v_pred + K * innovation;

        v_filt_ens(:,i) = v_ens;
        v_f_mean(i)     = mean(v_ens);
    end

    %% enkbs

    v_s_ens  = zeros(m_ens,N);
    v_s_ens(:,N) = v_filt_ens(:,N);

    v_s_mean = zeros(1,N);
    v_s_mean(N) = mean(v_s_ens(:,N));

    for i = N-1:-1:1
        u_i = u_truth(i+1);

        P_i = var(v_filt_ens(:,i+1),1);

        for j = 1:m_ens
            v_s  = v_s_ens(j,i+1);
            v_fi = v_filt_ens(j,i+1);
            dB_v = dW_v_ens(j,i);

            f_s = -d_v * v_s - c * u_i^2 + F_v;

            % backward step
            v_s_new = v_s ...
                - f_s * dt ...
                - sigma_v * dB_v ...
                - (sigma_v^2 / P_i) * (v_s - v_fi) * dt;

            v_s_ens(j,i) = v_s_new;
        end

        v_s_mean(i) = mean(v_s_ens(:,i));
    end

    %% RMSE

    err_f_enkbf = v_truth - v_f_mean;
    err_s_enkbs = v_truth - v_s_mean;

    rms_f_enkbf = sqrt(mean(err_f_enkbf.^2));
    rms_s_enkbs = sqrt(mean(err_s_enkbs.^2));
    
  
end
