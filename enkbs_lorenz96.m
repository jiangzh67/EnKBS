% Lorenz-96 EnKBF/EnKBS

set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');


rng(123); % for reproduction

%% Parameters

n  = 40;     % dimension
F  = 8;      % forcing
dt = 0.0005;  % Euler--Maruyama step size
T  = 100;    % total time window length
Nt = round(T/dt) + 1;          % number of time points
tt = (0:Nt-1) * dt;            % time vector


sigma_x2 = 5; % odd indices: larger noise
sigma_y2 = 0.1; % even indices: smaller noise
 
m_ens = 10; % ensemble size

%% Generate truth
% odd components use sigma_x2, even components use sigma_y2
sigma = sqrt(sigma_x2) * ones(n,1);       
sigma(2:2:n) = sqrt(sigma_y2);      

x_truth = zeros(n, Nt);

x0 = F * ones(n,1) + sqrt(0.1) * randn(n,1);
x_truth(:,1) = x0;

% Time marching
for k = 2:Nt
    x = x_truth(:,k-1);
    f = l96_drift(x, F);
    x_truth(:,k) = x + f*dt + sigma .* sqrt(dt) .* randn(n,1);
end

%% Reference heatmap
% downsample the truth trajectory for plotting
dt_plot = 0.05;     % different plotting time step           
skip = max(1, round(dt_plot/dt));
t_start = 1;
idx0 = find(tt >= t_start, 1, 'first');
idx_plot = idx0:skip:Nt;
tt_plot = tt(idx_plot);
x_plot  = x_truth(:, idx_plot);
figure;
imagesc(tt_plot, 1:n, x_plot);
axis xy;
box on;
set(gca,'fontsize',16);
xlabel('$t$');
title('Spatiotemporal patterns of Lorenz-96');
colorbar;
colormap(flipud(gray));
xlim([T - 25, T]);
ylim([1, n]);


%% main
r0    = 3; % localization radius
delta = sqrt(1.005); % inflation factor
rng(123);
[rmseF, rmseS] = l96_sweep_rmse( ...
            n, F, dt, Nt, m_ens, r0, sigma_x2, sigma_y2, delta, x_truth);

fprintf('r0=%g, delta^2=%g: RMSE_f=%.4f, RMSE_s=%.4f\n', ...
            r0, delta^2, rmseF, rmseS);

function [rmse_f, rmse_s] = l96_sweep_rmse(n, F, dt, Nt,  m_ens, r0, sigma_x2, sigma_y2, delta, x_truth)
%% even components are observed, odd components are hidden
tt = (0:Nt-1) * dt;            % time vector
idx_x = 1:2:n;   % hidden state indices (odd)
idx_y = 2:2:n;   % observed process indices (even)
Dim_x = numel(idx_x);
Dim_y = numel(idx_y);

y_obs = x_truth(idx_y,:);  % Dim_y x Nt

% Noise levels
sigma_x = sqrt(sigma_x2)   * ones(Dim_x,1);  
sigma_y = sqrt(sigma_y2) * ones(Dim_y,1);   

%% localization setup
% cyclic distances on the Lorenz-96 grid

[Ixx,Jxx] = ndgrid(idx_x, idx_x);   % odd-odd
D_xx = abs(Ixx - Jxx);
D_xx = min(D_xx, n - D_xx);

[Ixy,Jxy] = ndgrid(idx_x, idx_y);   % odd-even
D_xy = abs(Ixy - Jxy);
D_xy = min(D_xy, n - D_xy);

C_loc_xx = gaspari_cohn(D_xx / r0);
C_loc_xy = gaspari_cohn(D_xy / r0);


%% EnKBF
Gamma_inv = 1 / sigma_y2;

% initial hidden ensemble
x_ens = repmat(x_truth(idx_x,1), 1, m_ens) + 0.1*randn(Dim_x, m_ens);

% forward EnKBF storage
x_filt_ens = zeros(Dim_x, m_ens, Nt);
x_filt_ens(:,:,1) = x_ens;

x_mean = zeros(Dim_x, Nt);
x_std  = zeros(Dim_x, Nt);
x_mean(:,1) = mean(x_ens,2);
x_std(:,1)  = std(x_ens,0,2);

dB_x = sqrt(dt) * randn(Dim_x, m_ens, Nt-1);

dW_y = sqrt(dt) * randn(Dim_y, m_ens, Nt-1);

for k = 2:Nt
    y_prev = y_obs(:,k-1);
    y_curr = y_obs(:,k);
    dY     = y_curr - y_prev;

    % hidden drift and observed drift for each ensemble member
    f_vals = zeros(Dim_x, m_ens);
    z_vals = zeros(Dim_y, m_ens);

    for e = 1:m_ens
        x_full = zeros(n,1);
        x_full(idx_y) = y_prev;
        x_full(idx_x) = x_ens(:,e);

        drift_full = l96_drift(x_full, F);

        f_vals(:,e) = drift_full(idx_x);
        z_vals(:,e) = drift_full(idx_y);
    end

    x_bar = mean(x_ens,2);
    z_bar = mean(z_vals,2);

    X_anom = x_ens  - x_bar;
    Z_anom = z_vals - z_bar;
    P_xz = (X_anom * Z_anom') / (m_ens - 1);

    % localized Kalman gain
    P_xz_loc = C_loc_xy .* P_xz;
    K = P_xz_loc * Gamma_inv;

    % Euler--Maruyama forecast step
    x_pred = x_ens + f_vals*dt + (sigma_x .* dB_x(:,:,k-1));

    
    for e = 1:m_ens
        innovation = dY - z_vals(:,e)*dt + sigma_y .* dW_y(:,e,k-1);
        x_ens(:,e) = x_pred(:,e) + K * innovation;
    end

    
    % multiplicative inflation
    xbar_a = mean(x_ens,2);
    x_ens  = xbar_a + delta * (x_ens - xbar_a);
        
    x_filt_ens(:,:,k) = x_ens;
    x_mean(:,k) = mean(x_ens,2);
    x_std(:,k)  = std(x_ens,0,2);
end

%% EnKBS
% terminal condition: smoother equals filter at the final time

x_s_ens  = zeros(Dim_x, m_ens, Nt);
x_s_ens(:,:,Nt) = x_filt_ens(:,:,Nt);

x_s_mean = zeros(Dim_x, Nt);
x_s_std  = zeros(Dim_x, Nt);
x_s_mean(:,Nt) = mean(x_s_ens(:,:,Nt),2);
x_s_std(:,Nt)  = std(x_s_ens(:,:,Nt),0,2);

for k = Nt-1:-1:1

    Xf = x_filt_ens(:,:,k+1);
    xbar = mean(Xf,2);
    A = Xf - xbar;
    
    P = (A*A')/(m_ens - 1);    
    P_loc = C_loc_xx .* P;
    

    y_next = y_obs(:,k+1);

    for e = 1:m_ens
        xs = x_s_ens(:,e,k+1);
        xf = x_filt_ens(:,e,k+1);

        x_full = zeros(n,1);
        x_full(idx_y) = y_next;
        x_full(idx_x) = xs;

        drift_full = l96_drift(x_full, F);
        f_s = drift_full(idx_x);
        
        % reuse the same noise
        dB = dB_x(:,e,k);

        % backward attraction toward the filtered ensemble
        attract = (sigma_x.^2) .* (P_loc \ (xs - xf));

        x_s_ens(:,e,k) = xs  - f_s*dt  - sigma_x.*dB  - attract*dt;                     
    end

    x_s_mean(:,k) = mean(x_s_ens(:,:,k),2);
    x_s_std(:,k)  = std(x_s_ens(:,:,k),0,2);
end


%% Plot: x_1 truth vs filter/smoother

figure('Units','pixels','Position',[100 100 1600 420]);
hold on;

plot(tt, x_truth(1,:), 'k-', 'LineWidth', 1.2);

%% EnKBF
x1_f_mean = x_mean(1,:);
x1_f_std  = x_std(1,:);
plot(tt, x1_f_mean, 'r-', 'LineWidth', 1.2);
upper_f = x1_f_mean + 2*x1_f_std;
lower_f = x1_f_mean - 2*x1_f_std;
patch([tt, tt(end:-1:1)], [lower_f, upper_f(end:-1:1)], ...
      'r','FaceAlpha',0.35,'LineStyle','none');

%% EnKBS

x1_s_mean = x_s_mean(1,:);
x1_s_std  = x_s_std(1,:);

upper_s = x1_s_mean + 2*x1_s_std;
lower_s = x1_s_mean - 2*x1_s_std;


plot(tt, x1_s_mean, 'g-', 'LineWidth', 1.2);
patch([tt, tt(end:-1:1)], [lower_s, upper_s(end:-1:1)], ...
      'g','FaceAlpha',0.35,'LineStyle','none');

%% RMSE on a time window
% average over all hidden components and selected time
rmse_window = [20 100];  

if isempty(rmse_window)
    mask = true(size(tt));
else
    mask = (tt >= rmse_window(1)) & (tt <= rmse_window(2));
end
err_f = x_truth(idx_x,mask) - x_mean(:,mask);
err_s = x_truth(idx_x,mask) - x_s_mean(:,mask);

rmse_f = sqrt(mean(err_f(:).^2));
rmse_s = sqrt(mean(err_s(:).^2));

%% Figure cosmetics
box on;
grid on; grid(gca,'minor');
set(gca,'fontsize',16);
set(gca,'Position',[0.06 0.18 0.92 0.75]);

xlabel('$t$');
yl = ylabel('$x_1$');
set(yl,'Rotation',0);

title('$x_1$: Truth vs. filter/smoother' );
legend('Truth', 'Filter mean', 'Filter $\pm 2\sigma$', ...
       'Smoother mean', 'Smoother $\pm 2\sigma$', ...
       'Location','best','Orientation','horizontal');

xlim([75 100]);


%% Plot: filter/smoother std

figure('Units','pixels','Position',[120 160 1400 380]);
hold on;

plot(tt, x_std(1,:),   'r-', 'LineWidth', 1.5);
plot(tt, x_s_std(1,:), 'g-', 'LineWidth', 1.5);

box on;
grid on; grid(gca,'minor');
set(gca,'fontsize',16);
set(gca,'Position',[0.06 0.18 0.92 0.75]);

xlabel('$t$');
yl = ylabel('std');
set(yl,'Rotation',0);

title('$x_1$: filter/smoother standard deviation');
legend('Filter std', 'Smoother std', ...
       'Location','best','Orientation','horizontal');

xlim([90 100]);
end

function f = l96_drift(x, F)
    % Lorenz-96 drift with periodic indexing
    xp1 = circshift(x,-1);
    xm1 = circshift(x, 1);
    xm2 = circshift(x, 2);
    f = (xp1 - xm2) .* xm1 - x + F;
end

function w = gaspari_cohn(r)
% Gaspari--Cohn taper

    r = abs(r);
    w = zeros(size(r));

    m1 = (r < 1);
    rr = r(m1);
    w(m1) = 1 - (5/3)*rr.^2 + (5/8)*rr.^3 + (1/2)*rr.^4 - (1/4)*rr.^5;

    m2 = (r >= 1) & (r < 2);
    rr = r(m2);
    w(m2) = 4 - 5*rr + (5/3)*rr.^2 + (5/8)*rr.^3 - (1/2)*rr.^4 + (1/12)*rr.^5 - (2./(3*rr));

    % r>=2 already zero
end
