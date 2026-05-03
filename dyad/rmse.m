% nonlinear dyad model
% u is observed and v is hidden
close all;
clc;
clearvars;
    
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
    
rng(333) % for reproduction

%% parameters
N = 100000; % number of time steps
dt = 0.005; % time step
 
u_truth = zeros(1,N); % observed state
v_truth = zeros(1,N); % hidden state

Dim_v = 1; % hidden dimension

F_u = 1;
F_v = 0.8;

d_u = 0.5;
d_v = 0.5;

c = 2;

sigma_u = 0.5;
sigma_v = 1;


%% generate truth

for i = 2:N
    u_truth(i) = u_truth(i-1) + ( ( - d_u + c * v_truth(i-1)) * u_truth(i-1) + F_u) * dt + sqrt(dt) * randn * sigma_u;
    v_truth(i) = v_truth(i-1) + (- d_v * v_truth(i-1) - c * u_truth(i-1)^2 + F_v) * dt + sqrt(dt) * randn * sigma_v;
end

%% cgns filtering

mu_f = zeros(Dim_v,N); % filter mean
R_f = zeros(Dim_v.^2,N); % filter covariance

mu0 = zeros(Dim_v,1);
R0 = eye(Dim_v)*0.01;
R_f(:,1) = reshape(R0,Dim_v.^2,1);

for i = 2:N
    u0 = u_truth(i-1);
    u = u_truth(i);
    a1 = - d_v;
    a0 = - c * u0^2 + F_v;
    A0 = -d_u * u0 + F_u;
    A1 = c * u0;
    Gamma_inv = 1 / sigma_u^2;

    % filter update
    mu = mu0 + (a0 + a1 * mu0) * dt + (R0 * A1') * Gamma_inv * (u - u0 - A0*dt - A1 * mu0 * dt);
    R = R0 + (a1 * R0 + R0 * a1' + sigma_v^2 - (R0 * A1') * Gamma_inv * (R0 * A1')') * dt;     
    
    mu_f(:,i) = mu;
    R_f(:,i) = R;

    mu0 = mu;
    R0 = R;

end



%% cgns smoothing

mu_s = zeros(Dim_v,N); % smoother mean
R_s = zeros(Dim_v.^2,N); % smoother covariance
 
mu_s(:,end) = mu;
R_s(:,end) = reshape(R,Dim_v.^2,1);

rng(2);
randn(3,N);

for i = N-1:-1:1
    u0 = u_truth(:,i);
    a1 = - d_v;
    a0 = - c * u0^2 + F_v;    
    R_filter = reshape(R_f(:,i),Dim_v,Dim_v);
    A = eye(Dim_v) + a1 * dt;
    K_s = R_filter * A' * (sigma_v^2 * dt + A * R_filter * A')^(-1);
    mu_s(:,i) = mu_f(:,i) + K_s * (mu_s(:,i+1) - a0 * dt - A * mu_f(:,i));
    R_next = reshape(R_s(:,i+1),Dim_v,Dim_v);
    R_next = R_filter + K_s * (R_next - A * R_filter * A' - sigma_v^2 * dt) * K_s';   
    R_s(:,i) = reshape(R_next,Dim_v.^2,1);
end



%% cgns RMSE

err_f_cg = v_truth - mu_f;
rms_f_cg = sqrt(mean(err_f_cg.^2));

err_s_cg = v_truth - mu_s;
rms_s_cg = sqrt(mean(err_s_cg.^2));

%% enkbf/enkbs RMSE for different ensemble sizes

m_list = [4,5,6,8,10,15,20,50,100,200,500,1000,2000];
nM = numel(m_list);

rms_f_enkbf_list = zeros(1,nM);
rms_s_enkbs_list = zeros(1,nM);

for k = 1:nM
    m_ens = m_list(k);
    
  
    [rms_f_enkbf, rms_s_enkbs] = run_enkf_enks_dyad( ...
        m_ens, dt, N, ...
        u_truth, v_truth, ...
        d_u, d_v, c, F_u, F_v, sigma_u, sigma_v);
    
    rms_f_enkbf_list(k) = rms_f_enkbf;
    rms_s_enkbs_list(k) = rms_s_enkbs;
end


%% plot RMSE vs ensemble size

figure;
hold on;
h_f  = plot(m_list, rms_f_enkbf_list, '-or', 'LineWidth', 1.5);
h_s  = plot(m_list, rms_s_enkbs_list, '-sg', 'LineWidth', 1.5);
h_cf = yline(rms_f_cg, 'r--', 'LineWidth', 2);
h_cs = yline(rms_s_cg, 'g--', 'LineWidth', 2);
hold off;
box on; grid on; grid(gca,'minor');
set(gca,'FontSize',16);
set(gca,'XScale','log');   

xticks([4 8 20 40 80 150 300 500 1000 2000]);
xticklabels({'4','8','20','40','80','150','300','500','1000','2000'});
xlim([4,2000]);

xlabel('$m$','Interpreter','latex');
title('RMSE for filter/smoother','Interpreter','latex');

legend([h_f,h_s,h_cf,h_cs], ...
       {'EnKBF', 'EnKBS','Optimal filter','Optimal smoother'}, ...
       'Location','best');

