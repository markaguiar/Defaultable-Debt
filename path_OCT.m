%path_sanfran_500.m
%first run sanfran
sim=100;
smooth=1600;
nos=500;
T=10000;

for s=1:sim
   

   
%generate path of z:
%shockz will be uniform between 0 and 1
%temp will be the cdf of z_t given z_t-1
%prob(temp(z(i-1))<shockz<=temp(z(i))) = pdf(z(i))


shockz=rand(T,1); %shockz is uniform between 0 and 1
zpath=zeros(T,1);
zpath(1,1)=find(z==max(z(find(abs(muz-z)==min(abs(muz-z)))))); %start zpath at z closest to muz

for t=2:T;
    temp=cumsum(pdfz(zpath(t-1),:));
    if shockz(t)<temp(1);
        zpath(t,1)=1;
    elseif shockz(t)>temp(Z);
        zpath(t,1)=Z;
    else
        zpath(t,1)=find(shockz(t)<=temp(2:Z) & shockz(t)>temp(1:Z-1))+1; 
    end;
end;



%generate path of g -- same procedure as for z:

shockg=rand(T,1); 
gpath=zeros(T,1);
gpath(1,1)=round(G/2); 

for t=2:T;
    temp=cumsum(pdfg(gpath(t-1),:));
    if shockg(t)<temp(1);
        gpath(t,1)=1;
    elseif shockg(t)>temp(G);
        gpath(t,1)=G;
    else
        gpath(t,1)=find(shockg(t)<=temp(2:G) & shockg(t)>temp(1:G-1))+1; 
    end;
end;



%generate redemption realization (1=redemption):

redemp = ones(floor(lambda*T),1);
redemp = [redemp; zeros(T-length(redemp),1)];
temp = randperm(T)';
redemp = redemp(temp);

%starting values
apath=zeros(T,1);
defpath=zeros(T,1);
history=zeros(T,1);
state=zeros(T,1);
qpath=zeros(T,1);
Edefpath=ones(T,1);

apath(1)=azero;

for t=1:T-1;
    state(t) = (gpath(t)-1)*A*Z+ (zpath(t)-1)*A + apath(t);
    if history(t)==0 | (history(t)==1 & redemp(t)==1);
        if default(state(t))==0;
            apath(t+1)=policygood(state(t));
            qpath(t) = q0(state(t),policygood(state(t)));
            Edefpath(t)=Edef(state(t),policygood(state(t)));
            Vpath(t)=Vgood(state(t));
        elseif default(state(t))==1;
            apath(t+1)=azero;
            history(t+1)=1;
            defpath(t)=1;
            qpath(t) = 1/(1+rbase);
            Edefpath(t)=0;
            Vpath(t)=Vbad(state(t));


        end;
    elseif history(t)==1 & redemp(t)==0;
            apath(t+1)=azero;
            history(t+1)=1;
            qpath(t) = 1/(1+rbase);
            Edefpath(t)=0;
            Vpath(t)=Vbad(state(t));

    end;
end;
  
qpath(T)=NaN;

if endow==0
    labpath = ((1-alpha)*g(gpath)/mug.*exp(z(zpath))).^(1/(alpha+omega-1));
else
    labpath=ones(T,1);
end
ypath = exp(z(zpath)).*(labpath.^(1-alpha)).*g(gpath)/mug; 
ay=a(apath(2:T))./(ypath(1:T-1));

logy = log(ypath)+cumsum(log(g(gpath)))-log(g(gpath))+log(mug);
assets = a(apath).*exp(cumsum(log(g(gpath)))-log(g(gpath))+log(mug));
dassets = assets(2:T)-assets(1:T-1);
dassetsy = dassets./exp(logy(1:T-1));
nx = assets(2:T).*qpath(1:T-1)-assets(1:T-1);
nx(find(defpath(1:T-1)==1))=0;
c = exp(logy(1:T-1)) - nx;
c(find(defpath(1:T-1)==1))=exp(logy(find(defpath(1:T-1)==1)));

sovr = 1./qpath - 1;
spread = sovr-rbase;


sample = defpath==0 & (history==0 | redemp==1);


sovrt=sovr(T-nos:T-1);
logyt=logy(T-nos:T-1);
nxt = nx(T-nos:T-1);
nxy = nxt./exp(logyt);
lconst = log(c(T-nos:T-1));
spreadt=spread(T-nos:T-1);
spreadtannual=(1+spreadt).^4-1;
labt=log(labpath(T-nos:T-1));

hp_spread=hpfilter(spreadt,smooth);
hp_spreadtannual=spreadtannual-hpfilter(spreadtannual,1600);
hp_y = hpfilter(logyt,smooth);
hp_nx = hpfilter(nxy,smooth);
hp_sovr=hpfilter(sovrt,smooth);
hp_cons = hpfilter(lconst,smooth);
hp_lab=hpfilter(labt,smooth);

ydev=logyt-hp_y;
nxdev = nxy-hp_nx;
sovdev=sovrt- hp_sovr;
consdev=lconst-hp_cons;
spreadev=spreadt-hp_spread;
labdev=labt-hp_lab;



% note that the spread in the model is quarterly while it is annual in the
% data. need to make appropriate adjustment
STD=std([ydev, spreadev, nxdev, consdev]);
CC=corrcoef([ydev, spreadev, nxdev, consdev]);
AC=corrcoef(ydev(1:end-1),ydev(2:end));
stdyrnxc(s,:)=STD;
ccr(s,:)=[CC(1,2:4),CC(2,3),AC(1,2)];
defaultpc(s)=mean(defpath(find(apath<azero)))*100;

end

disp('mean std y r nx c')
disp(mean(stdyrnxc,1))
disp('mean correlation yr ynx yc rnx yy')
disp(mean(ccr,1))
disp('mean default')
disp(mean(defaultpc))