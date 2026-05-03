close all;
clc;
clearvars;
    
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
    
seed_truth = 333;
rng(seed_truth) %for repeating experiments

% Parameters
N = 20000+1; % total number of numerical integration time steps
dt = 0.001; % numerical integration time step

% Ensemble size for EnKBF/EnKBS
m_ens = 50;
Dim_v = 1; % hidden dimension

% Model parameters
F_u = 1;
F_v = 0.8;
d_u = 0.5;
d_v = 0.5;
c = 2;
sigma_u = 0.5;
sigma_v = 1;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Generating the truth signal %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_full = 300000;
u_full = zeros(1,N_full);
v_full = zeros(1,N_full);
u_full(1) = 0.9;
v_full(1) = -0.2;

for i = 2:N_full
    u_full(i) = u_full(i-1) + ( ( - d_u + c * v_full(i-1)) * u_full(i-1) + F_u) * dt + sqrt(dt) * randn * sigma_u;
    v_full(i) = v_full(i-1) + (- d_v * v_full(i-1) - c * u_full(i-1)^2 + F_v) * dt + sqrt(dt) * randn * sigma_v;
end

t_start = 180.1;
t_end   = 200.1;
idx_truth = round(t_start/dt) : round(t_end/dt);

u_truth = u_full(idx_truth);
v_truth = v_full(idx_truth);
rng(seed_truth);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Filtering %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

v_cg_f_mean = zeros(Dim_v,N); % filter mean
v_cg_f_cov = zeros(Dim_v.^2,N); % filter covariance

mu0 = zeros(Dim_v,1);
R0 = eye(Dim_v)*0.01;
v_cg_f_mean(:,1) = mu0;
v_cg_f_cov(:,1) = reshape(R0,Dim_v.^2,1);

for i = 2:N
    u0 = u_truth(i-1);
    u = u_truth(i);
    a1 = - d_v;
    a0 = - c * u0^2 + F_v;
    A0 = -d_u * u0 + F_u;
    A1 = c * u0;
    Gamma_inv = 1 / sigma_u^2;

    % cgns filter update
    mu = mu0 + (a0 + a1 * mu0) * dt + (R0 * A1') * Gamma_inv * (u - u0 - A0*dt - A1 * mu0 * dt);
    R = R0 + (a1 * R0 + R0 * a1' + sigma_v^2 - (R0 * A1') * Gamma_inv * (R0 * A1')') * dt;     
    
    v_cg_f_mean(:,i) = mu;
    v_cg_f_cov(:,i) = R;

    mu0 = mu;
    R0 = R;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% CG TRIANGLE TABLE (naive) %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

v_cg_mean_tri = NaN(N, N); % rows: time, cols: terminal time
v_cg_cov_tri  = NaN(N, N);

a1  = -d_v;
phi = (1 + a1*dt);
Q   = sigma_v^2 * dt;

for k = 1:N
    n = k;

    mu_next = v_cg_f_mean(:, n);
    R_next  = v_cg_f_cov(:, n);

    v_cg_mean_tri(k, k) = mu_next;
    v_cg_cov_tri(k, k)  = R_next;

    for i = (n-1):-1:1
        R_filter = v_cg_f_cov(:, i);

        K_s = (R_filter * phi) / (Q + (phi*phi)*R_filter);

        u_left = u_truth(i);
        a0_i   = -c * (u_left^2) + F_v;

        mu_f_i = v_cg_f_mean(:, i);
        mu_i   = mu_f_i + K_s * (mu_next - a0_i*dt - phi*mu_f_i);

        R_i = R_filter + (K_s*K_s) * (R_next - (phi*phi)*R_filter - Q);

        mu_next = mu_i;
        R_next  = R_i;

        v_cg_mean_tri(i, k) = mu_i;
        v_cg_cov_tri(i, k)  = R_i;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% EnKBF / EnKBS for dyad (u observed, v hidden) %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial ensemble for v (around the true initial v)
v_ens = v_truth(1) + 0.5 * randn(m_ens,1);

% Store forward EnKBF ensemble
v_filt_ens  = zeros(m_ens,N);
v_filt_ens(:,1) = v_ens;

% Store Brownian increments for v-ensemble (for reuse in EnKBS)
dW_v_ens = sqrt(dt) * randn(m_ens,N-1);

% Store Brownian increments for perturbed observations (stochastic EnKBF)
dW_u_obs = sqrt(dt) * randn(m_ens,N-1);

% Forward means and variances
v_f_mean = zeros(1,N);
v_f_var  = zeros(1,N);
v_f_mean(1) = mean(v_ens);
v_f_var(1)  = var(v_ens,0);

% ----- EnKBF forward loop -----
for i = 2:N
    u_prev = u_truth(i-1);
    u_curr = u_truth(i);
    dY     = u_curr - u_prev;  % observation increment du

    % ensemble statistics at previous time
    v_bar = mean(v_ens);

    % drift f(v,u) and observation drift h(v,u)
    f_vals = -d_v * v_ens - c * u_prev^2 + F_v;           % hidden drift
    h_vals = (-d_u + c * v_ens) * u_prev + F_u;           % observation drift

    h_bar = mean(h_vals);

    % ensemble covariance: P^{vh}
    dv = v_ens - v_bar;
    dh = h_vals - h_bar;
    P_vh = (dv' * dh) / (m_ens - 1);    % scalar cross-covariance

    % Kalman-type gain (scalar obs noise variance Gamma = sigma_u^2)
    K = P_vh / (sigma_u^2);

    % Brownian increments for v
    dB_v = dW_v_ens(:,i-1);

    % Forecast step for v
    v_pred = v_ens + f_vals*dt + sigma_v * dB_v;

    innovation = dY - h_vals*dt + sigma_u * dW_u_obs(:,i-1);

    v_ens = v_pred + K * innovation;

    % store
    v_filt_ens(:,i) = v_ens;
    v_f_mean(i)     = mean(v_ens);
    v_f_var(i)      = var(v_ens,0);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% EnKBS TRIANGLE TABLE (naive) %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% triangle storage on the time grid
% rows: smoothed time, cols: terminal time, only rows <= cols used
v_enkbs_mean_tri = NaN(N, N);
v_enkbs_var_tri  = NaN(N, N);

for k = 1:N
    n = k;                 % terminal index for this column

    % terminal condition at t_n: smoother = filter at n
    v_s = v_filt_ens(:, n);         % m_ens x 1

    v_enkbs_mean_tri(k, k) = mean(v_s);
    v_enkbs_var_tri(k, k)  = var(v_s, 0);

    % backward integration
    for i = (n-1):-1:1
        u_right = u_truth(i+1);
        P_right = var(v_filt_ens(:, i+1), 0);
        
        % forward Brownian increment on [t_i, t_{i+1}]
        dB_v = dW_v_ens(:, i);       % m_ens x 1

        % drift evaluated at current smoother state
        f_s = -d_v * v_s - c * (u_right^2) + F_v;   % m_ens x 1

        % backward step
        v_s = v_s ...
              - f_s * dt ...
              - sigma_v * dB_v ...
              - (sigma_v^2 / P_right) * (v_s - v_filt_ens(:, i+1)) * dt;
    
        v_enkbs_mean_tri(i, k) = mean(v_s);
        v_enkbs_var_tri(i, k)  = var(v_s, 0);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% ACI SURFACE for Objective CIR %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% delta(j,n) = KL( N(mu_full(j), R_full(j)) || N(mu_part(j,n), R_part(j,n)) )
% 1D Gaussian KL:
% 0.5 * [ (mu_full - mu_part)^2 / R_part + (R_full/R_part) - 1 - log(R_full/R_part) ]

delta_CG    = NaN(N, N);
delta_EnKBS = NaN(N, N);


% Loop over columns (terminal observation time)
for k = 1:N
    rset = 1:k;  % only j<=n valid

    % ---------------- CG: full smoother vs partial smoother ----------------
    mu_full = v_cg_mean_tri(rset, N).';    % 1 x k
    R_full  = v_cg_cov_tri( rset, N).';    % 1 x k
    mu_part = v_cg_mean_tri(rset, k).';    % 1 x k
    R_part  = v_cg_cov_tri( rset, k).';    % 1 x k
    
    cov_ratio = R_full ./ R_part;
    
    sig_term  = 0.5 * ((mu_full - mu_part).^2) ./ R_part;
    disp_term = 0.5 * (cov_ratio - 1 - log(cov_ratio));
    delta_CG(rset, k)        = (sig_term + disp_term).';

    % ---------------- EnKBS: full smoother vs partial smoother -------------
    mu_full = v_enkbs_mean_tri(rset, N).';   % 1 x k
    R_full  = v_enkbs_var_tri( rset, N).';   % 1 x k
    mu_part = v_enkbs_mean_tri(rset, k).';   % 1 x k
    R_part  = v_enkbs_var_tri( rset, k).';   % 1 x k

    cov_ratio = R_full ./ R_part;

    sig_term  = 0.5 * ((mu_full - mu_part).^2) ./ R_part;
    disp_term = 0.5 * (cov_ratio - 1 - log(cov_ratio));
    delta_EnKBS(rset, k)        = (sig_term + disp_term).';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Objective CIR (approx) from delta_EnKBS table %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   delta_EnKBS : N x N upper-triangular,
%                delta_EnKBS(r,k) = KL( N(mu_full(r),R_full(r)) || N(mu_{r|k},R_{r|k}) ), r<=k

approx_obj_CIR_EnKBS = zeros(1, N);
RE_metric_threshold = 1e-5;                % same spirit as paper

for r = 1:N
    % delta as a function of lagged observation time T' (k=r..N)
    RE_n = delta_EnKBS(r, r:N);
    
    maxRE = max(RE_n);

    % If essentially zero, set CIR=0 to avoid numerical inflation
    if maxRE <= RE_metric_threshold
        approx_obj_CIR_EnKBS(r) = 0;
        continue;
    end
    
    I = simps(RE_n) * dt;      % simps with unit spacing
    approx_obj_CIR_EnKBS(r) = I / maxRE;
end


tt = (0:N-1) * dt;                           % full time axis (consistent with u_truth length N)
figure('Position',[100 100 1500 520]);   % long and narrow
subplot(3,1,1)
plot(tt, u_truth, 'm', 'LineWidth', 2);
hold on;
plot(tt, v_truth, 'b', 'LineWidth', 2);
yline(d_u/c, 'k--', 'LineWidth', 2);

grid on; box on;
set(gca, 'FontSize', 16);
title(sprintf('EnKBS m = %d: Time series for $v \\rightarrow u$', m_ens));
legend('$u$', '$v$', '$d_u/c$','NumColumns', 4, 'Location', 'best');
set(gca,'XGrid', 'on', 'YGrid', 'off')
h=ylabel('$u,v$');
h.Rotation = 0;
h.Units = 'normalized';
pos = h.Position;
pos(1) = -0.04;
h.Position = pos;

h.HorizontalAlignment = 'right';
h.VerticalAlignment   = 'middle';

% CIR
subplot(3,1,2);
plot(tt, approx_obj_CIR_EnKBS, 'k-', 'LineWidth', 1.6);
grid on; box on;
set(gca,'FontSize',16);
h=ylabel('CIR');
h.Rotation = 0;
h.Units = 'normalized';
pos = h.Position;
pos(1) = -0.04;
h.Position = pos;

h.HorizontalAlignment = 'right';
h.VerticalAlignment   = 'middle';

subplot(3,1,3)
ACI_curve = diag(delta_EnKBS);       % N x 1
plot(tt, ACI_curve, 'k-', 'LineWidth', 2);
grid on; box on;
set(gca,'FontSize',16);
h=ylabel('ACI');
h.Rotation = 0;
xlabel('$t$');
ylim([0,4]);
h.Units = 'normalized';
pos = h.Position;
pos(1) = -0.04;
h.Position = pos;

h.HorizontalAlignment = 'right';
h.VerticalAlignment   = 'middle';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Objective CIR (approx) from delta_CG table %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   delta_CG : N x N upper-triangular,
%              delta_CG(r,k) = KL( N(mu_full(r),R_full(r)) || N(mu_{r|k},R_{r|k}) ), r<=k

approx_obj_CIR_CG = zeros(1, N);

RE_metric_threshold = 1e-5;            % same spirit as paper

for r = 1:N
    % delta as a function of lagged observation time T' (k=r..N)
    RE_n = delta_CG(r, r:N);

    maxRE = max(RE_n);

    % If essentially zero, set CIR=0 to avoid numerical inflation
    if maxRE <= RE_metric_threshold
        approx_obj_CIR_CG(r) = 0;
        continue;
    end

    I = simps(RE_n) * dt;      % simps with unit spacing
    approx_obj_CIR_CG(r) = I / maxRE;
end

tt = (0:N-1) * dt;
figure('Position',[100 100 1500 520]); 
subplot(3,1,1);
plot(tt, u_truth, 'm', 'LineWidth', 2);
hold on;
plot(tt, v_truth, 'b', 'LineWidth', 2);
yline(d_u/c, 'k--', 'LineWidth', 2);

grid on; box on;
set(gca, 'FontSize', 16);
title('CG: Time series for $v \rightarrow u$');
h = ylabel('$u,v$');
h.Rotation = 0;
legend('$u$', '$v$', '$d_u/c$','NumColumns', 4, 'Location', 'best');
set(gca,'XGrid', 'on', 'YGrid', 'off')
h.Units = 'normalized';
pos = h.Position;
pos(1) = -0.04;
h.Position = pos;

h.HorizontalAlignment = 'right';
h.VerticalAlignment   = 'middle';

% CIR
subplot(3,1,2);
plot(tt, approx_obj_CIR_CG, 'k-', 'LineWidth', 1.6);
grid on; box on;
set(gca,'FontSize',16);
h = ylabel('CIR');
h.Rotation = 0;
h.Units = 'normalized';
pos = h.Position;
pos(1) = -0.04;
h.Position = pos;

h.HorizontalAlignment = 'right';
h.VerticalAlignment   = 'middle';

subplot(3,1,3);
% curve from the final terminal time
ACI_curve = diag(delta_CG);        % N x 1
plot(tt, ACI_curve, 'k-', 'LineWidth', 2);
grid on; box on;
set(gca,'FontSize',16);
h = ylabel('ACI');
h.Rotation = 0;
xlabel('$t$'); 
ylim([0,4]);
h.Units = 'normalized';
pos = h.Position;
pos(1) = -0.04;
h.Position = pos;

h.HorizontalAlignment = 'right';
h.VerticalAlignment   = 'middle';





