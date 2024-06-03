clear all;
m = 1;
q = -1;
L = 2*pi;
th = 0.5;
E_func = @(t) [-m/q*sin(t) + m/(2*q)*cos(t), -m/(2*q)*sin(t), -m/(2*q)*sin(t) - 3*m/(2*q)*cos(t)]';
B_func = @(t) [-m/(2*q), m/(2*q),m/(2*q)]';
info.E_func = E_func;
info.B_func = B_func;
analytical_x = @(t) [sin(t)]';
analytical_v = @(t) [cos(t),-sin(t),cos(t) - sin(t)]';

k = 3;
omega = k;
analytical_E = @(x,t) [0,cos(omega*t)*sin(k*x),0]';
analytical_B = @(x,t) [0,0, -sin(omega*t)*cos(k*x)]';

t0 = 0;
t_end = 1;
Np = 1;
Nx_base = 10;
Nt_base = 10;
refinements = 5;

t = t0;
Nx = 10;
dx = L/Nx;
E_start = zeros(Nx,3);
B_start = zeros(Nx,3);
E_end = zeros(Nx,3);
B_end = zeros(Nx,3);

info.t0 = t0;
info.t_end = t_end;
info.Nx = Nx;
info.Np = Np;
info.L = L;
info.dx = dx;
info.th = th;
info.qom = q/m;

for i = 1:Nx
    x = (i-1)*dx;
    E_start(i,:) = analytical_E(x,t0);
    B_start(i,:) = analytical_B(x + dx/2,t0);
    E_end(i,:) = analytical_E(x,t_end);
    B_end(i,:) = analytical_B(x + dx/2,t_end);
end

errors = zeros(refinements,4);
convergence = zeros(refinements,4);
for i = 1:refinements
    NT = Nt_base*2^(i-1);
    dt = (t_end-t0)/NT;
    info.dt = dt;
    info.NT = NT;
    x_start = analytical_x(t0-dt/2);
    x_end = analytical_x(t_end-dt/2);
    v_start = analytical_v(t0);
    v_end = analytical_v(t_end);
    xp = x_start;
    vp = v_start;
    E0 = E_start;
    Bc = B_start;
    [xp,vp,E0,Bc] = solveECSIM(xp,vp,E0,Bc, info);
    time_errors(i,:) = [norm(xp-x_end)/norm(x_end),norm(vp-v_end)/norm(v_end),norm(E0-E_end)/norm(E_end),norm(Bc-B_end)/norm(B_end)];
    if(i>1)
        time_convergence(i,:) = time_errors(i,:)./time_errors(i-1,:);
    end
end
time_errors
time_convergence
%%
info.t0 = t0;
info.t_end = t_end;
NT = 200;
dt = (t_end-t0)/NT;
info.dt = dt;
info.NT = NT;
info.Np = Np;
info.L = L;
info.th = th;
info.qom = q/m;
x_start = analytical_x(t0-dt/2);
x_end = analytical_x(t_end-dt/2);
v_start = analytical_v(t0);
v_end = analytical_v(t_end);
errors = zeros(refinements,4);
convergence = zeros(refinements,4);
for i = 1:refinements
    Nx = Nx_base*2^(i-1);
    dx = L/Nx;
    info.Nx = Nx;
    info.dx = dx;
    E_start = zeros(Nx,3);
    B_start = zeros(Nx,3);
    E_end = zeros(Nx,3);
    B_end = zeros(Nx,3);
    for j = 1:Nx
        x = (j-1)*dx;
        E_start(j,:) = analytical_E(x,t0);
        B_start(j,:) = analytical_B(x + dx/2,t0);
        E_end(j,:) = analytical_E(x,t_end);
        B_end(j,:) = analytical_B(x + dx/2,t_end);
    end

    xp = x_start;
    vp = v_start;
    E0 = E_start;
    Bc = B_start;
    [xp,vp,E0,Bc] = solveECSIM(xp,vp,E0,Bc, info);
    space_errors(i,:) = [norm(xp-x_end)/norm(x_end),norm(vp-v_end)/norm(v_end),norm(E0-E_end)/norm(E_end),norm(Bc-B_end)/norm(B_end)];
    if(i>1)
        space_convergence(i,:) = space_errors(i,:)./space_errors(i-1,:);
    end
end
space_errors
space_convergence


function [xp,vp,E0,Bc] = solveECSIM(xp,vp,E0,Bc,info)
    Ex0 = E0(:,1);
    Ey0 = E0(:,2);
    Ez0 = E0(:,3);
    Byc = Bc(:,2);
    Bzc = Bc(:,3);
    t = info.t0;
    for it=1:info.NT
        for ip=1:info.Np
            Bp = info.B_func(t);
            alphap(ip,1:3,1:3)=alpha(info.qom*info.dt/2,Bp(1),Bp(2),Bp(3));
        end
        Jx0=zeros(info.Nx,1);
        Jy0=zeros(info.Nx,1);
        Jz0=zeros(info.Nx,1);
        M=zeros(info.Nx,info.Nx,3,3);
    
        un=ones(info.Nx,1);
    
        Derv=spdiags([-un un],[-1 0],info.Nx,info.Nx); Derv(1,info.Nx)=-1; Derv=Derv*info.dt*info.th/info.dx;
    
        Derc=spdiags([-un un],[0 1],info.Nx,info.Nx); Derc(info.Nx,1)=1; Derc=Derc*info.dt*info.th/info.dx;
    
        bKrylov=[Ex0-Jx0*info.dt*info.th; Ey0-(Jy0)*info.dt*info.th; Ez0-(Jz0)*info.dt*info.th; Byc; Bzc];
        
        I = sparse(eye(info.Nx));
        O = sparse(zeros(info.Nx));
        Maxwell=[ I, O, O, O O;
            O, I, O, O,Derv;
            O, O, I,  -Derv,  O;
            O,O,   -Derc, I, O;
            O,Derc, O,O,I];
    
        xKrylov=Maxwell\bKrylov;
    
        Ex12=xKrylov(1:info.Nx);
        Ey12=xKrylov(info.Nx+1:2*info.Nx);
        Ez12=xKrylov(2*info.Nx+1:3*info.Nx);
        Byc=(xKrylov(3*info.Nx+1:4*info.Nx)-Byc*(1-info.th))/info.th;
        Bzc=(xKrylov(4*info.Nx+1:5*info.Nx)-Bzc*(1-info.th))/info.th;
    
        Ex0=(Ex12-Ex0*(1-info.th))/info.th;
        Ey0=(Ey12-Ey0*(1-info.th))/info.th;
        Ez0=(Ez12-Ez0*(1-info.th))/info.th;

        E0 = [Ex0,Ey0,Ez0];
        Bc = [Bc(:,1),Byc,Bzc];
        xp=xp+vp(1,:)*info.dt;
        for ip=1:info.Np
            vp(:,ip) =(2*reshape(alphap(ip,:,:),3,3)-eye(3))*vp(:,ip) + info.dt*info.qom*reshape(alphap(ip,:,:),3,3)*info.E_func(t+info.th*info.dt); 
            
        end
        t = t+info.dt;
    end
end