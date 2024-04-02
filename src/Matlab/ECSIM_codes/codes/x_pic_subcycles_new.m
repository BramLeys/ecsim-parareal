%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% ECsim 1D MATLAB Implementation
% 1D-1V Electrostatic formulation based on Valsov-Ampere
% MIT Licencse: Giovanni Lapenta, KULeuven
%
% Two Stream Instability 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clear all

graphics = true

CFL=[];
DxoLde=[];
Efluct=[];
Kfluct=[];
vth=[];
Enerr=[];
MomErr=[];

%Sampling in x,y cooridnates
VT=.02; % thermische snelheid (relatief tov c)
V0=.1; % netto speed (relatief tov c)
VB=0*2; % basis speed voor particles added to thermic speed
V1=.1*V0; % perturbatieschalering voor mode

for Nsub=1:10

    Nsub

    told = cputime

    implicit_moment_limit=false; ninner=1;

    Nx=64;
    L=2*pi;
    dx=L/Nx;

    T=50; T=15
    NTOUT=50;
    CFL=.1*Nsub
    dt=CFL*dx/V0;
    NT=T/dt;
    qom=-1; % electic charge mass density of particle

    Np=100000;
    dx=L/Nx;

    xv=linspace(0,L,Nx+1)'; % grid points for Electric field
    xc=.5*(xv(2:end)+xv(1:end-1))'; % grid points for Magnetic field
    Vx=dx*ones(Nx,1); % volume of each cell
    Ev0=zeros(Nx,1);
    mode=5;


    xp=linspace(0,L-L/Np,Np)';  % original position of particles
    up=VB+VT*randn(Np,1);   % original thermal velocity of particles
    sigma_x=.01;

    pm=[1:Np]';
    pm=1-2*mod(pm,2);   % used to define the two different streams (one going one direction the other the other direction)
    up=up+pm.*V0;   % Create the two opposing streams

    up=up+V1*sin(2*pi*xp/L*mode);  % add some disturbance of the chosen mode


    %computing overlap
    ix=1+floor(xp/dx); % cell of the particle, first cell is cell 1, first node is 1 last node Nx+1
    frac1 = 1-(xp/dx-ix+1); % how far is the particle from the vertex of the cell (procent)
    ix2=mod(ix,Nx)+1; % defines the second cell that has influence from the previous cell to local support
                      % of b-spline of order 1 being larger than 1 cell
                      % length

    M=zeros(Nx,Nx);
    for ip=1:Np
        M(ix(ip),ix(ip))=M(ix(ip),ix(ip))+frac1(ip).^2/2; %divided by 2 to reflect the matrix to symmetrize
        M(ix2(ip),ix(ip))=M(ix2(ip),ix(ip))+frac1(ip).*(1-frac1(ip));
        M(ix2(ip),ix2(ip))=M(ix2(ip),ix2(ip))+(1-frac1(ip)).^2/2;
    end
    M=M+M';

    rhotarget=exp(-(xv(1:Nx)/L).^2/sigma_x^2);
    rhotarget=-1;
    rhotarget=rhotarget.*Vx;
    rhotildeV=M\rhotarget;



    for ip=1:Np
        qp(ip)=rhotildeV(ix(ip))*frac1(ip)+rhotildeV(ix2(ip))*(1-frac1(ip));
    end

   

    %
    %   Setting initial rho
    %

    rho=zeros(Nx,1);
    for ip=1:Np
        rho(ix(ip))=rho(ix(ip))+frac1(ip)*qp(ip);
        rho(ix2(ip))=rho(ix2(ip))+(1-frac1(ip))*qp(ip);
    end
    rho=rho./Vx;
    rhotarget=rhotarget./Vx;
    rho_c = .5*(rho(2:end)+rho(1:end-1));rho_c=0;

    if(graphics)
        figure(1)
        subplot(2,1,1)
        plot(xv(1:Nx),rho,xv(1:Nx),rhotarget,'o')
        subplot(2,1,2)
        plot(qp)
    end

    % assign initial velocity

    JV0= VB * rhotarget;

    histEnergy = [];
    histEnergyP = [];
    histEnergyK = [];
    histMomentum = [];
    hE =[];
    histTime=[];

    time_start = clock();
    for it=1:NT



        J0=zeros(Nx,1);


        M=zeros(Nx,Nx);
        ix=zeros(Nsub,Np);
        ix2=zeros(Nsub,Np);
        frac1=zeros(Nsub,Np);
        fraction_p=zeros(Np,Nx);
        for itsub=1:Nsub
            xp=xp+up*dt/Nsub; %update position

            xp=mod(xp,L); % periodic boundaries



            ix(itsub,:)=1+floor(xp'/dx); % cell of the particle, first cell is cell 1, first node is 1 last node Nx+1
            frac1(itsub,:) = 1-(xp'/dx-ix(itsub,:)+1);
            ix2(itsub,:)= mod(ix(itsub,:),Nx)+1; % second cell of influence


            for ip=1:Np
                fraction_p(ip,ix(itsub,ip)) = fraction_p(ip,ix(itsub,ip)) +frac1(itsub,ip)/Nsub;
                fraction_p(ip,ix2(itsub,ip)) = fraction_p(ip,ix2(itsub,ip)) +(1-frac1(itsub,ip))/Nsub;
            end

        end

        for ip=1:Np
            J0 = J0 + fraction_p(ip,:)'*qp(ip)*up(ip)/dx; % does not add the dt_v/dt factor (probably since in this case it reduces to 1 anyways)
        end

        if(implicit_moment_limit) %It requires only one subcycle
            rho=zeros(Nx,1);
            for ip=1:Np
                rho(ix(ip))=rho(ix(ip))+frac1(ip)*qp(ip);
                rho(ix2(ip))=rho(ix2(ip))+(1-frac1(ip))*qp(ip);
            end
            M=diag(rho);
            
        else

            for ip=1:Np
                M=M+fraction_p(ip,:)'*fraction_p(ip,:)*qp(ip);
            end
        end

        un=ones(Nx,1);
        Ampere=spdiags([un],[0],Nx,Nx)+qom*dt^2/4*M/dx;
        Ev12=Ampere\(Ev0-(J0-JV0)*dt/2);

        %
        % GMRES Below works about just as well as the direct solver. Ampere is very
        % diagonally dominant and GMRES works very well because of the condition
        % number. However the overall energy conservation is not as perfect becasue
        % of the tolerance in the iterative Krylov solver.
        %
        %Ev12=gmres(Ampere,Ev0-J0*dt/2,10,1e-7,100);

        Ev0=2*Ev12-Ev0;

        Jbar = J0 + M*Ev12*qom*dt/2/dx;
        rho_c = rho_c - dt * (Jbar(2:end)-Jbar(1:end-1))/dx;

        if(graphics)
            figure(3)
            subplot(3,1,1)
            plot(xv(2:end-1),rho_c)
            ylabel('\rho')
            pause(.1)
            subplot(3,1,2)
            divE_c=(Ev0(2:end)-Ev0(1:end-1))/dx;
            ylabel('\nabla \cdot E')
            plot(xv(2:end-1),divE_c)
            subplot(3,1,3)
            plot(xv(2:end-1),divE_c-rho_c)
            ylabel('\nabla \cdot E-\rho')
            xlabel('x')
        end

        upold=up;
        if(implicit_moment_limit)
            ubar=up;
            for inner=1:ninner
                for ip=1:Np
                    ubar(ip)= up(ip)+0.5*dt*(Ev12(ix(ip))*frac1(ip)+Ev12(ix2(ip))*(1-frac1(ip)))*qom;
                    xbar(ip)= xp(ip) + 0.5*dt*ubar(ip);
                    xbar(ip)=mod(xbar(ip),L);
                    ix(ip)=1+floor(xbar(ip)/dx); % cell of the particle, first cell is cell 1, first node is 1 last node Nx+1
                    frac1(ip) = 1-(xbar(ip)/dx-ix(ip)+1);
                    ix2(ip)=mod(ix(ip),Nx)+1;
                end
            end
            up=2*ubar -up;
        else
            for itsub=1:Nsub
                for ip=1:Np
                    up(ip)=up(ip)+dt/Nsub*(Ev12(ix(itsub,ip))*frac1(itsub,ip)+Ev12(ix2(itsub,ip))*(1-frac1(itsub,ip)))*qom;
                end
            end
        end

        Ek = 0.5*sum(qp'.*up.^2/qom);
        Ep = 0.5*sum(Ev0.^2)*dx;
        Etot = Ek + Ep;
        histEnergy = [histEnergy Etot];
        histEnergyP = [histEnergyP Ep];
        histEnergyK = [histEnergyK Ek];
        histMomentum = [histMomentum qp*up./sum(qp)];
        histTime = [histTime dt*it];

        hE = [hE Ev12];


        if(mod(it,round(NT/NTOUT))==0&graphics)
            figure(2)
            subplot(2,3,1:2)
            plot(xp,up,'.')
            xlim([0 L])
            title(['\omega_{pe}t = ' num2str(it*dt) '   CFL = ' num2str(std((up))*dt/dx) '    Dx/\lambda_{De}= ' num2str(dx/VT) ])
            subplot(2,3,3)
            semilogy(dt*(1:it),histEnergyP)
            subplot(2,3,4)
            plot(1:it,histEnergy-histEnergy(1))
            subplot(2,3,5)
            plot(1:it,histEnergyP,1:it,histEnergyK,1:it,histEnergy)
            subplot(2,3,6)
            plot(1:it,histMomentum)
            axis tight
            pause(.1)
        end


    end
    % timing
    time_end = clock();
    time_elapsed = etime(time_end,time_start)

    t=dt*(1:it);
    gamma=.35;
    if(graphics)
        figure(100)
        subplot(2,3,1:2)
        plot(xp,up,'.')
        title(['\omega_{pe}t = ' num2str(it*dt) '   CFL = ' num2str(mean(abs(up))*dt/dx) '    Dx/\lambda_{De}= ' num2str(dx/VT) ])
        subplot(2,3,3)
        semilogy(t,histEnergyP,t(1:end/2),1e-5*exp(2*gamma*t(1:end/2)))
        subplot(2,3,4)
        plot(1:it,histEnergy-histEnergy(1))
        subplot(2,3,5)
        plot(1:it,histEnergyP,1:it,histEnergyK,1:it,histEnergy)
        subplot(2,3,6)
        plot(1:it,histMomentum/Np./mean(abs(qp'.*up)))
        axis tight
        pause(.1)
    end

    save ECIM histEnergy histEnergyP histEnergyK xp up time_elapsed;

    Enerr=[Enerr; mean(abs(histEnergy-histEnergy(1)))/histEnergy(1)];
    CFL=[CFL; std(up)*dt/dx];
    DxoLde=[DxoLde;  dx/std(up)];
    Efluct=[Efluct; std(histEnergyP(round(NT/2):end)/histEnergy(1))];
    Kfluct=[Kfluct; std(histEnergyK(round(NT/2):end)/histEnergy(1))];
    vth=[vth; std(up)];
    MomErr=[MomErr; std(diff(histMomentum)/dt/std(up)/sum(qp))];


    tnew = cputime;
    duration = tnew -told

    disp('saving')
    namefile=['caso' num2str(Nsub)];
    save(namefile, 'histEnergy', 'histEnergyP', 'histEnergyK', 'histMomentum', 'histTime', 'hE', 'duration', 'xp','up')
    told=cputime

end

figure
loglog(DxoLde,vth,DxoLde,Enerr,DxoLde,Efluct,DxoLde,Kfluct,DxoLde,MomErr)
legend('V_{th}','<\Delta E_t>/E_t(0)','\deltaE_E/E_t(0)','\deltaE_K/E_t(0)','\nu/\omega_{pe}','location','SouthWest')
xlabel('\Delta x/\lambda_{De}','fontsize',[15])
set(gca,'fontsize',[15])
xlim([min(DxoLde) max(DxoLde)]);
print -depsc lapenta_elst_fg


