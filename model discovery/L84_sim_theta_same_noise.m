function [x, y, z] = L84_sim_theta_same_noise(dt, N, Theta, sig, x0, y0, z0, dW)
% simulate l84 in the learned library form with shared noise
% inputs: dt scalar, N scalar
%         Theta 36x1
%         sig 3x1
%         x0, y0, z0: scalars
%         dW: 3 x (N-1)
% outputs: x, y, z: 1xN

    assert(numel(Theta)==36, 'Theta must be 36x1.');
    assert(numel(sig)==3, 'sig must be 3x1.');
    assert(all(size(dW)==[3, N-1]), 'dW must be 3 x (N-1).');

    th = parseTheta36_L84(Theta);

    x = zeros(1,N); y = zeros(1,N); z = zeros(1,N);
    x(1)=x0; y(1)=y0; z(1)=z0;

    sx = sig(1); sy = sig(2); sz = sig(3);

    for k = 2:N
        yk = y(k-1); zk = z(k-1); xk = x(k-1);

        phi = basis_phi_L84(yk, zk);

        a0x = th.x0.' * phi;  a1x = th.x1.' * phi;
        a0y = th.y0.' * phi;  a1y = th.y1.' * phi;
        a0z = th.z0.' * phi;  a1z = th.z1.' * phi;

        fx = a0x + a1x * xk;
        fy = a0y + a1y * xk;
        fz = a0z + a1z * xk;

        x(k) = xk + fx*dt + sx * dW(1,k-1);
        y(k) = yk + fy*dt + sy * dW(2,k-1);
        z(k) = zk + fz*dt + sz * dW(3,k-1);
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
