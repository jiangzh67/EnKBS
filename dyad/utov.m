close all;
clc;
clearvars;
    
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
    
seed_truth = 333;
rng(seed_truth) % fix the random number seed for repeating experiments

% Parameters
N = 20000+1; % total number of numerical integration time steps
dt = 0.001; % numerical integration time step

% Ensemble size for EnKBF/EnKBS
m_ens = 50;

% inflation factor
delta_infl2 = 1;
delta_infl= sqrt(delta_infl2);

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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% EnKBF / EnKBS for dyad (v observed, u hidden) %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% initial hidden ensemble
u_ens = u_truth(1) + 0.1 * randn(m_ens,1);

% forward EnKBF storage
u_filt_ens  = zeros(m_ens,N);
u_filt_ens(:,1) = u_ens;

dW_u_ens = sqrt(dt)*randn(m_ens,N-1);

dW_v_obs = sqrt(dt)*randn(m_ens,N-1);
   
u_f_mean = zeros(1,N);
u_f_var  = zeros(1,N);
u_f_mean(1) = mean(u_ens);
u_f_var(1)  = var(u_ens,0);

for i = 2:N
    v_prev = v_truth(i-1);
    v_curr = v_truth(i);
    dY     = v_curr - v_prev;

    u_bar = mean(u_ens);

    % hidden drift and observed drift
    f_vals =  (-d_u + c * v_prev) * u_ens + F_u;
    h_vals =  -d_v * v_prev - c * (u_ens.^2) + F_v;

    h_bar = mean(h_vals);

    du = u_ens - u_bar;
    dh = h_vals - h_bar;
    P_uh = (du' * dh) / (m_ens - 1);

    K = P_uh / (sigma_v^2);

    dB_u = dW_u_ens(:,i-1);

    % forecast step
    u_pred = u_ens + f_vals*dt + sigma_u * dB_u;

    innovation = dY - h_vals*dt + sigma_v * dW_v_obs(:,i-1);

    u_ens = u_pred + K * innovation;
    
    u_filt_ens(:,i) = u_ens;
    u_f_mean(i)     = mean(u_ens);
    u_f_var(i)      = var(u_ens,0);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% EnKBS TRIANGLE TABLE (naive) %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% triangle storage on the time grid
% rows: smoothed time, cols: terminal time, only rows <= cols used
u_enkbs_mean_tri = NaN(N, N);
u_enkbs_var_tri  = NaN(N, N);

for k = 1:N
    n = k;                 % terminal index for this column

    % terminal condition at t_n: smoother = filter at n
    u_s = u_filt_ens(:, n);         % m_ens x 1

    % store at (n,n)
    u_enkbs_mean_tri(k, k) = mean(u_s);
    u_enkbs_var_tri(k, k)  = var(u_s, 0);

    % backward integration
    for i = (n-1):-1:1
        v_right = v_truth(i+1);
        P_right = var(u_filt_ens(:, i+1), 0);

        % forward Brownian increment on [t_i, t_{i+1}]
        dB_u = dW_u_ens(:, i);       % m_ens x 1

        % drift evaluated at current smoother state
        f_s = (-d_u + c * v_right) * u_s + F_u;   % m_ens x 1

        % backward step
        u_s = u_s ...
              - f_s * dt ...
              - sigma_u * dB_u ...
              - (sigma_u^2 / P_right) * (u_s - u_filt_ens(:, i+1)) * dt;
        
        u_enkbs_mean_tri(i, k) = mean(u_s);
        u_enkbs_var_tri(i, k)  = var(u_s, 0);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% ACI SURFACE for Objective CIR %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% delta(j,n) = KL( N(mu_full(j), R_full(j)) || N(mu_part(j,n), R_part(j,n)) )
% 1D Gaussian KL:
% 0.5 * [ (mu_full - mu_part)^2 / R_part + (R_full/R_part) - 1 - log(R_full/R_part) ]

delta_EnKBS = NaN(N, N);


% Loop over columns (terminal observation time)
for k = 1:N
    rset = 1:k;  % only j<=n valid

    % ---------------- EnKBS: full smoother vs partial smoother -------------
    mu_full = u_enkbs_mean_tri(rset, N).';   % 1 x k
    R_full  = u_enkbs_var_tri( rset, N).';   % 1 x k
    mu_part = u_enkbs_mean_tri(rset, k).';   % 1 x k
    R_part  = u_enkbs_var_tri( rset, k).';   % 1 x k

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


%% plot
tt = (0:N-1) * dt;                           % full time axis (consistent with u_truth length N)
figure('Position',[100 100 1500 520]);   % long and narrow
subplot(3,1,1)
plot(tt, u_truth, 'm', 'LineWidth', 2);
hold on;
plot(tt, v_truth, 'b', 'LineWidth', 2);
yline(d_u/c, 'k--', 'LineWidth', 2);
grid on; box on;
set(gca, 'FontSize', 16);
title(sprintf('EnKBS m = %d: Time series for $u \\rightarrow v$', m_ens));
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



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%% Err/Std (EnKBF/EnKBS) %%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



enkbs_mean = u_enkbs_mean_tri(1:N, N).';   % 1-by-k_plot
enkbs_var  = u_enkbs_var_tri(1:N,  N).';   % 1-by-k_plot
enkbs_std  = sqrt(enkbs_var);             % guard tiny negative
% bands (mean ± 2σ)
enkbs_upper = enkbs_mean + 2*enkbs_std;
enkbs_lower = enkbs_mean - 2*enkbs_std;


figure;
% (a) truth
subplot(3,1,1);
hold on;
plot(tt, u_truth,     'm-', 'LineWidth', 2);
plot(tt, v_truth,     'b-', 'LineWidth', 2);
hold off;
box on; grid on; grid(gca,'minor');
set(gca,'fontsize',16);
legend('$u$', '$v$', ...
       'Location','best','Orientation','horizontal');
title('Truth');
% (b) truth

% Filter RMS
err_f_enkbf = u_truth - u_f_mean;
rms_f_enkbf = sqrt(mean(err_f_enkbf.^2));
% Smoother RMS
err_s_enkbs = u_truth - enkbs_mean;
rms_s_enkbs = sqrt(mean(err_s_enkbs.^2));
subplot(3,1,2);
hold on;
plot(tt, err_f_enkbf, 'r-', 'LineWidth',2);
plot(tt, err_s_enkbs, 'g-', 'LineWidth', 2);
yline(0, 'k--', 'LineWidth', 2);
hold off;
box on; grid on; grid(gca,'minor');
set(gca,'fontsize',16);
legend('Filter', 'Smoother', '0',...
       'Location','best','Orientation','horizontal');
title('Error');

% (c) std 
sigma_f_enkbf = sqrt(u_f_var);
subplot(3,1,3);
hold on;
plot(tt, sigma_f_enkbf, 'r-', 'LineWidth', 2);
plot(tt, enkbs_std, 'g-', 'LineWidth', 2);
hold off;
box on; grid on; grid(gca,'minor');
set(gca,'fontsize',16);
xlabel('t');
title('Std');
legend('Filter','Smoother', 'Location','best','Orientation','horizontal');





