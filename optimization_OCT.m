%baseline_g.m
%baseline model with growth shocks

%In detrended form:
%endowment:  y=exp(z)*g/mu_g
%preferences  u(c)=c^(1-s)/(1-s)
%if labor supply endogenous:
%y=exp(z)*g/mu_g*L^(1-alpha)
%u(c,L)=1/(1-s)*(c-1/omega*L^omega)^(1-s)

%states:    z -- log productivity -- space is vector of length Z with
%               intervals delz
%           g -- log trend growth rate -- space is vector of legnth G with
%               intervals delg
%           a -- financial assets -- space is vector of length A
%                with intervals dela (a = 0 at azero). negative means debt
%           h -- credit history (good,  bad)

%parameters
%           lambda -- prob of transiting from h=bad to h=good
%           s -- coeff of rel risk aversion
%           cost -- additional output cost if in autarky
%           beta -- discount factor
%           alpha -- capital share of output
%           abound -- limit of bailouts
%           omega -- labor supply parameter (if endow==0)
%           rbase is world riskfree rate
%           beta is time preference (if uzawa==1, use uzawa preferences and
%           endogenize time preference)

%distributions:  productivity parameters: z is AR(1) with long rum mean muz, AR coef rhoz, innovation
%           standard deviation sdz, and number of states Z, 
%           g is AR(1) with mug, rhog, sdg and number of states G
%           transition matrices have rows are t and col are t+1
%           pdfz -- transition matrix for z (Z x Z)
%           underlying parameters:
%           (i) z is log normal with long run mean muz, stdev sdz, and ar coeff of rhoz
%               z_t+1 = (1-rhoz)*muz + rhoz*z_t + u_t+1, u dist
%               normal(0,sdz)

%functions:     q is the pricing vector for domestic debt.  It has size
%                   A*Z*G x A.  For a given row (a,z,g), q is the
%                   inverse interest rate if we choose column (a')
%               Vgood is value of having good credit rating given a,z,g.
%                   It has size AxZxG
%               Vbad is value of having bad credit rating given a,z,g.
%                   It has size AxZxG.  Note that a does not affect Vbad as a
%                   always equals zero under bad credit history
%               Vbadgood is value of having good credit history but zero
%                   assets (value after transiting out of bad credit history)
%               policygood and policybad is index of choice of (a') given
%                   state (a,z,g).  Has size (A*Z*G,1)
%
%
%Outline:  1. Define parameters
%          2.  Calculate transition matrices for z and g
%          3.  Calculate transition matrix for pair (z,g)
%          4. Initial state vectors and value functions
%          5.  Start interest rate iteration:  updates interest rate as
%              function of state (start at q=1/(1+r*)
%          6.  Value function interation:  takes q (interest rate) as given and iterates until Vgood and Vbad converge.
%          7.  Update interest rate given new value functin and corresponding prob of default




clear all;


%Define parameters:
alpha=0.32; %share of capital in output (or 1-share of labor) -- used if endow==0
lambda = 0.1; %prob of redemption 0.1
s = 2; %crra parameter
sdz=.034;%.034; %stdev of log prod shock
muz=-0.5*sdz^2;; %long run mean of log prod
rhoz=0.9;%mean reversion of log prod
rhoz_alt=0.9; %used to define state space
Z=1; %25 %number of states for log prod
A=400 %500; %number of states for assets 
rbase=0.01; %baseline interest rate
psi=0.11; %time preference parameter if using Uzawa preferences
cost=0.02;%0.02; %pct of output forgone in autarky 0.02
omega=1.455; %elasticity of labor supply parameter
G=25; %number of states for growth rate
sdg=.03;%0.03
mug=1.006;
rhog=0.17; %0.17
rhog_alt=0.17; %used to define state space
abound=0; %remember minus sign
zbase=0; %value of z if Z==1 (i.e. only trend shocks)
betabase = 0.8;%discount rate
uzawa=0; %set to 1 if using Uzawa preferences
endow=1; %if endow=1, endowment; otherwise endow=0 and labor supply is endogenous

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%set up state space 

%limits of asset space
    amax=0.0;
    if Z>1;
        amin=-.3;%-.27; 
    elseif (G>1 & Z==1) | abound<0;
        amin=-0.22; %-0.22
    end;

    if cost<=0.005;
        amin=-.065;
    end;
    
%grid for assets:

dela = (amax-amin)/(A-1);
a=amin:dela:amax;
a(find(a==min(a(a>=0))))=0; %ensures zero is a state
a=a(:);
azero = find(a==0); %identifies where a==0


%state space and transition matrix for z:

if Z==1;
    
    z=zbase;
    pdfz=1;
    
else;
   
lrsdz=sdz/(1-rhoz_alt^2)^(.5); %stdev of invariant distribution of log prod
m=2.5; %how "wide" discretization of z state space (how many standard deviations on each side of lr mean)
delz=2*m/(Z-1)*lrsdz; %length of intervals between z in discretizations
z=muz-m*lrsdz:delz:muz+m*lrsdz;
z=z(:);

%prob of z transiting from i to j: F(z_j+dz/2|i)-F(z_j-dz/2|i) where F is
%normal cdf with mean (1-rhoz)muz+rhoz*z_i and stdev sdz:

for i=1:Z;
    for j=1:Z;
        pdfz(i,j)=normcdf(z(j)+delz/2,(1-rhoz)*muz+rhoz*z(i),sdz)-normcdf(z(j)-delz/2,(1-rhoz)*muz+rhoz*z(i),sdz);
    end;
end;


pdfz=pdfz.*((1./(sum(pdfz,2)))*ones(1,Z)); %ensures each row adds up to one across columns

end;


%state space and transition matrix for g:
if G==1;
    g=mug;
    pdfg=1;
  else;
   
lrsdlogg=sdg/(1-rhog_alt^2)^(.5) %stdev of invariant distribution of log g
lrsdg=(exp(2*log(mug)+2*(lrsdlogg)^2)-exp(2*log(mug)+(lrsdlogg)^2))^(0.5)
m=4.1458; %how "wide" discretization of g state space (how many standard deviations on each side of lr mean)keep this wider because rhog is low.need to get the extreme draws.
delg=2*m/(G-1)*lrsdg; %length of intervals between g in discretizations
dellogg=2*m/(G-1)*lrsdlogg; %length of intervals between g in discretizations
g=mug-m*lrsdg:delg:mug+m*lrsdg;
gold=mug-m*lrsdlogg:dellogg:mug+m*lrsdlogg;
mugtest=exp(log(mug));
lrsdgtest=(exp(2*(log(mug)-0.5*lrsdlogg^2)+2*(lrsdlogg)^2)-exp(2*(log(mug)-0.5*lrsdlogg^2)+(lrsdlogg)^2))^(0.5)
g=g(:);
delgtest=2*m/(G-1)*lrsdgtest;
gtest=mugtest-m*lrsdgtest:delgtest:mugtest+m*lrsdgtest;
gtest=gtest(:);
g=gtest;

for i=1:G;
    for j=1:G;
        pdfg(i,j)=logncdf(g(j)+delg/2,(1-rhog)*(log(mug)-0.5*lrsdlogg^2)+rhog*log(g(i)),sdg)-logncdf(g(j)-delg/2,(1-rhog)*(log(mug)-0.5*lrsdlogg^2)+rhog*log(g(i)),sdg);
    end;
end;


pdfg=pdfg.*((1./(sum(pdfg,2)))*ones(1,G)); %ensures each row adds up to one across columns

end;  
    

%transition matrix for (z,g):
%Prob z=z_j,  g=g_n given z=z_i, g=g_m = pdfz(i,j)*pdfg(m,n)
%Note that pdfzg has size Z*G x Z*G.  
pdfzg=kron(pdfg,pdfz);
pdfzg=pdfzg./(((sum(pdfzg,2)))*ones(1,Z*G)); 




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialize value functions and interest rate function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initial guess for q:  initial q is the risk free interest rate
%current state characterized by vector of length A*Z*G -- note order!
%choices for next period a' from vector of length A
q0 = 1/(1+rbase)*ones(A*Z*G,A);


%inital guess for V: initial guess is all ones
Vgood = ones(A*Z*G,1); %good credit rating
Vbad = ones(A*Z*G,1); %bad credit rating (autarky)

%Vbadgood picks out value of being in good credit standing with zero assets (redemption)
Vbadgood = reshape(Vgood,A,Z*G);
Vbadgood = Vbadgood(azero,:); 
Vbadgood = ones(A,1)*Vbadgood;
Vbadgood = reshape(Vbadgood,A*Z*G,1);


%labor (used if endow==0):
% lab = ((1-alpha)*exp(z)*g)^(1/(omeag+alpha-1))
lab=(1-alpha)*(kron(ones(G,1),exp(z)).*kron(g,ones(Z,1)))/mug;
lab=lab.^(1/(omega+alpha-1));
lab=kron(lab,ones(A,1)); 
lab=(1-endow)*lab+endow*ones(size(lab)); %if endow=0, then lab=1


%current income:
y= kron(ones(G,1),exp(z)).*kron(g,ones(Z,1))/mug;
y = kron(y,ones(A,1)).*(endow+(1-endow)*(lab.^(1-alpha))); % add A dimension Now row 2 is first z, 2nd a, etc. A dimension is "first" dimension

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Begin interest rate iteration:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
diffq=1;
tolq=1e-6;

while diffq>tolq;


%Savings given choice of a'
%Note that future assets are discounted by price q.
%In detrended form, S=g*a'*q-a
S = (kron(g,ones(A*Z,1))*a').*q0-(kron(ones(Z*G,1),a)*ones(1,A));

%current consumption as function of current state and next period choice of
%a' without default (A*Z,A) 
c = y*ones(1,A) - S;

%current consumption as function of current state with default (consume endowment -- net of "cost")
cdefault = (1-cost)*y*ones(1,A);


%calculate utility:
%u is A*Z*G x A.  The columns map out the utility of choosing next
%period's a' given state today (row).
    x=(c-(1-endow)*(lab.^(omega)*(1/omega))*ones(1,A));
    xdef=(cdefault-(1-endow)*(lab.^(omega)*(1/omega))*ones(1,A));
if s~=1; %s is inverse of elasticity of substition
    u = (((x).^(1-s)))./(1-s);
    u(find(x<=0))=NaN; %eps is a very small number used by matlab (see help eps).  I use this in case c is negative
    udefault = (((xdef).^(1-s)))./(1-s);
    udefault(find(xdef<=eps))=NaN;
elseif s==1;
    u = log(max(x,realmin));
    u(find(x<=realmin))=NaN;
     udefault = log(max(xdef,eps));
     udefault(find(xdef<=eps))=NaN;
end;

%endogenous discount rate

if uzawa==1;
 beta=(kron(g,ones(A*Z,A))).^(1-s).*exp(-psi*(log(1+x)));
 beta(find(x<=0))=0;
 betabad=(kron(g,ones(A*Z,A))).^(1-s).*exp(-psi*(log(1+xdef)));
 betabad(find(xdef<=0))=0;
else;
    beta=(kron(g,ones(A*Z,A))).^(1-s)*betabase; %the g^(1-s) reflects we are in detrended form
    betabad=beta;   
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Value function iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Begin value function iteration:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
diff=10;
tolV = min(max(diffq,tolq),1e-6); %tighten tolV as diffq declines

while diff>tolV;


%calculate expected value given current state and choice of a and k:

EVgood = pdfzg*(reshape(Vgood,A,Z*G)'); %The last two "dimensions" are Z,G.
%Here Vgood will be next period's V assuming no default and the choice of
%next period's a'  (across the columns).
%Reshape puts z,g across cols and then the transpose puts them down the
%rows and a across the columns.  
%Multiplying by pdfzg gives the expected value of V given the row of
%pdfzg (current z,g) and the choice of a next period (across the
%columns)

EVgood = kron(EVgood,ones(A,1)); %This adds the current a across the rows.  

%same for EVbad which is V assuming default
EVbad = pdfzg*(reshape(Vbad,A,Z*G)');
EVbad = kron(EVbad,ones(A,1));

%EVbadgood is the value of having zero assets but the option to borrow or
%save (reinstatement of a good credit rating)
EVbadgood = pdfzg*(reshape(Vbadgood,A,Z*G)');
EVbadgood = kron(EVbadgood,ones(A,1));

%value functions
%For each one we take the max across next period's a' (the columns).
%We first solve for Vbad and then Vgood and define default as the states
%where Vbad>Vgood
%policygood and policybad are the respective cols where the max is located

[Vbad1,policybad] = max(udefault + betabad.*lambda.*EVbadgood + betabad.*(1-lambda).*EVbad,[],2);
[Vgood1,policygood] = max(u + beta.*EVgood,[],2);
default=Vbad1>Vgood1 | isnan(Vgood1)==1;
Vgood1(find(Vbad1>Vgood1 | isnan(Vgood1)==1))=Vbad1(find(Vbad1>Vgood1| isnan(Vgood1)==1));

%extract Vbadgood -- extract the value from Vgood of having a good credit
%rating but zero assets.
Vbadgood = reshape(Vgood1,A,Z*G);
Vbadgood = Vbadgood(azero,:);
Vbadgood = ones(A,1)*Vbadgood;
Vbadgood = reshape(Vbadgood,A*Z*G,1);

diff=max([max(max(abs(Vgood-Vgood1))),max(max(abs(Vbad-Vbad1)))]) %Check for convergence of value functions

Vgood=Vgood1;
Vbad=Vbad1;

end; %end value function iteration

%calculate q1:

%Expected default given current z,g and choice a':
Edef = pdfzg*(reshape(default,A,Z*G)'); 

ind = reshape(default,A,Z*G)';
ind = sum(ind)==Z*G; %ind finds a' in which default happens with prob 1 next period

for j=1:A;
    if ind(j)==1;
        Edef(:,j)=1;
    end;
end;

Edef = kron(Edef,ones(A,1)); %add current a to rows to make same size as q


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Adjust q for potential bailouts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if abound<0;
    aastar=a/abound;
    aastar(find(aastar==0))=eps;
    aastar=1./aastar;
else;
    aastar=zeros(A,1);
end;
    aastar(find(aastar>1))=1;
    aastar=ones(A*Z*G,1)*aastar';

q1=1/(1+rbase)*(aastar+(1-Edef).*(1-aastar));


q1=max(q1,0);


diffq = max(max(abs(q1-q0))) % check for convergence of q

q0=q1;

end; %end interest rate iteration

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%policy functions
%policygood and policybad have current state as row and optimal choice (position) of
%(a) as the value. 

policy=(1-default).*policygood+default.*azero;

%calculate x_t(state)
for i=1:A*Z*G;
    xt(i,1)=x(i,policy(i));
    qt(i,1)=q0(i,policy(i));
end;



if G>1;
figure(2);
defspace=abs(reshape(default,A,G));
mesh(a,g,defspace');
axis([min(a),max(a),min(g),max(g),0,1]);
view(0,90);
colormap(flipud(bone));
figure(1);
plot(a,q0(1,:),a,q0(end,:));


elseif Z>1;
    figure(2);
mesh(a,z,abs(reshape(default,A,Z))');
axis([min(a),max(a),min(z),max(z),0,1]);
view(0,90);
colormap(flipud(bone));
figure(1);
plot(a,q0(1,:),a,q0(end,:));

end;
