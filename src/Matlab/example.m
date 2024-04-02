clear all; clc; close all;
% Viscous burger: du/dt + u*du/dx = v*ddu/ddx
t_end = 0.08;
x_end = 1;
nb_proc = 64;
N_x_fine = 512;
dx_fine = x_end/N_x_fine;
N_x_coarse = 256;
dx_coarse = x_end/N_x_coarse;
dt = t_end/nb_proc;
nu = 0.005;
f_E =@(u,du) -u.*du;
f_I = @(u,ddu)nu.*ddu;
Lobatto_fine = 5;
Lobatto_coarse = 3;
SDC_sweeps = 2;
sigma = 0.004;
u0 = @(x) exp(-(x-0.5).^2/sigma);
du0 = @(x) -2/sigma * (x-0.5).*exp(-(x-0.5).^2/sigma);
ddu0 = @(x) -2/sigma.*exp(-(x-0.5).^2./sigma) -2/sigma^2.* (x-0.5).*exp(-(x-0.5).^2./sigma);


Lobatto_matrix_fine = kron(eye(nb_proc),[1/10,(49/90),(32/45),(49/90),(1/10)]);
Lobatto_matrix_coarse = kron(eye(nb_proc),[(1/3),(4/3),(1/3)]);
%% Solution variables
U = zeros(nb_proc,N_x_fine);
F = zeros(nb_proc,N_x_fine);

U_coarse = zeros(nb_proc,N_x_coarse);
F_coarse = zeros(nb_proc,N_x_coarse);

%% Initialization
for i = 1:nb_proc
    U(i,:) = u0((1:N_x_fine)*dx_fine);
    F(i,:) = f_E(U(i,:),du0((1:N_x_fine).*dx_fine)) + f_I(U(i,:),ddu0((1:N_x_fine).*dx_fine));
end
U_coarse = coarsen(U);
F_coarse = coarsen(F);

U_SDC = repmat(U,Lobatto_fine,1);
F_SDC = repmat(F,Lobatto_fine,1);

U_coarse_SDC = repmat(U_coarse,Lobatto_coarse,1);
F_coarse_SDC = repmat(F_coarse,Lobatto_coarse,1);

tau = dt*(Lobatto_matrix_coarse*F_coarse_SDC - coarsen(Lobatto_matrix_fine*F_SDC));

dt_1 = 1-sqrt(3/7);
dt_2 = sqrt(3/7);
for n = 1:nb_proc
    for j = 1:n
        U_tilde = 
    end
end


function B = coarsen(A)
    B = (A(:,1:2:end)+A(:,2:2:end))/2;
end