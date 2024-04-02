function [sp] = spettrale(spettro,dt,dx,nome_var)
global Nsm
[Nt Nx]=size(spettro')
h=figure(100)
set(h,'Position', [473 223 348 574])
subplot(2,1,1)
imagesc( [0 Nx*dx], [0 Nt*dt],spettro')
title(['XT plane ' nome_var])
xlabel('x/d_e','FontSize',14)
ylabel('\omega_{pe} t','FontSize',14)
set(gca,'FontSize',14)
axis xy
colorbar
subplot(2,1,2)
%
% FFT variable in time
%
Fs = 1/dt;                    % Sampling frequency
T = 1/Fs;                     % Sample time
w = Fs*pi*linspace(0,1,Nt/2+1);
%
% FFT variable in space
%
Fs = 1/dx;                    % Sampling frequency
T = 1/Fs;                     % Sample length
k = Fs*pi*linspace(0,1,Nx/2+1);

sp=fft2(spettro');
%pcolor(k,w,abs(sp(1:Nt/2+1,1:Nx/2+1)))
imagesc(k,w,log(abs(sp(1:Nt/2+1,1:Nx/2+1))))
%imagesc(k,w,log(abs(sp(1:Nt/2+1,:))))
axis xy
ylabel('\omega/\omega_{pe}','FontSize',14)
xlabel('kd_e','FontSize',14)
set(gca,'FontSize',14)
title(['Spectrum ' nome_var])
colorbar
%shading interp
load gist_ncar
colormap(gist_ncar)
clim([-8 3])
print(['spettro_' num2str(Nsm) '_' nome_var],'-dpng')
%close(100)
end

