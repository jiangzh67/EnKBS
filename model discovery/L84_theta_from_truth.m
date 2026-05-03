function Theta = L84_theta_from_truth(a, b, f, g)
% phi(y,z) = [1; y; z; y^2; z^2; y*z]
% truth library coefficients for lorenz-84
% inputs: a, b, f, g : scalars
% output: Theta:  36x1

    theta_x0 = [a*f; 0; 0; -1; -1; 0];
    theta_x1 = [-a;  0; 0;  0;  0; 0];

    theta_y0 = [g;  -1; 0; 0; 0; 0];
    theta_y1 = [0;   1; -b; 0; 0; 0];

    theta_z0 = [0;   0; -1; 0; 0; 0];
    theta_z1 = [0;   b;  1; 0; 0; 0];

    Theta = [theta_x0; theta_x1; theta_y0; theta_y1; theta_z0; theta_z1];
end
