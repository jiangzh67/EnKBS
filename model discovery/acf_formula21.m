function [acf,lags] = acf_formula21(u, maxLag)
    % inputs: u is N x 1 or 1 x N, maxLag is a scalar
    % outputs: acf and lags are 1 x (maxLag+1)
    u0 = u(:) - mean(u(:));
    [c,lags] = xcorr(u0, maxLag, 'biased');
    acf = c(lags>=0) / c(lags==0);
    lags = lags(lags>=0);
end
