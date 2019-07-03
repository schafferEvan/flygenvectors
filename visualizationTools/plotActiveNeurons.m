
function plotActiveNeurons


% needs to be cleaned up.  This code visualizes rasters of activity, sorted
% according to any desired clustering, and visualizes the spatial
% footprints of every cell, color coded by the same clustering.


addpath(genpath( '~/Dropbox/_code/'))
expDir = '/Users/evan/Dropbox/_sandbox/sourceExtraction/good/running/';%_runAndFeed/';
expID = '180824_f3r1';%'190424_f3';%'190514_CrzDh44_f4';%'0824_f2r2_cur'; %'0828_f2r3'; %'0110'; %'0312_f4';

fromGreenCC = false;
if fromGreenCC
    load([expDir,expID,'/post_fromYcc.mat'])
    load([expDir,expID,'/F.mat'])
    matfileRaw = matfile([expDir,expID,'/F.mat']);
    matfilePost = matfile([expDir,expID,'/post_fromYcc.mat']);
    ccFlag = ' from Ycc';
else
    load([expDir,expID,'/post_fromRcc.mat'])
    load([expDir,expID,'/F_fromRed.mat'])
    matfileRaw = matfile([expDir,expID,'/F_fromRed.mat']);
    matfilePost = matfile([expDir,expID,'/post_fromRcc.mat']);
    ccFlag = ' from Rcc';
end
load([expDir,expID,'/Ysum.mat'])
load([expDir,expID,'/alignedBehavAndStim.mat'])
load([expDir,expID,'/alignedBehavSmooth.mat'])

[d1,d2,d3] = size(Ysum);


beh = behSmooth; %alignedBehavior.legVar; %
behNorm = zeros(size(beh));
%beh = smoothdata(beh,'movmean',101);
trialTmp = unique(trialFlag);
for j=1:length(trialTmp)
    behPiece = beh(trialFlag==trialTmp(j));
    cmax = quantile(behPiece,.999); %quantile(beh,.95); %max(beh);
    cmin = quantile(behPiece,.1); %quantile(beh,.05); %min(beh);
    behNormPiece = (behPiece-cmin)/(cmax-cmin);
    behNormPiece(behNormPiece>1)=1;
    behNormPiece(behNormPiece<0)=0;
    behNorm(trialFlag==trialTmp(j)) = behNormPiece;
end

% ampTh = 2000; %3000; %2000; % discard if max of trace is below this
% magTh = 1;  %discard if mean of dOO is greater than this (motion)
% minTh = 1; % discard if min is greater than this
% maxTh = 0.3; % discard if max is smaller than this
% rsqTh = 1;%0.95;
% rgccTh = 0.9; % discard units in which red and green are very correlated
% % d=goodData(rsq(isGood)<.95,:);[~,loc]=max(d,[],2);
% % [a,b]=sort(loc,'ascend');
% M = max(F,[],2);
% ampIsGood = M>ampTh;
% rgccIsGood = rgCorr<rgccTh;
% %rsqIsGood = rsq<rsqTh;
% rsqFull = ones(size(isGood));
% rsqFull(isGood) = rsq;
% rsqIsGood = ones(size(rsqFull)); %rsqFull<rsqTh;
% % data=YgoodData(rsqIsGood,:);%note: this is equivalent to Fsc(goodIds,:)
% % goodIdsSNR = find(isGood);
% % goodIds = goodIdsSNR(rsqIsGood);
dYYfull = dYY; %zeros(size(F));
%dYYfull(find(oIsGood),:) = dYY;
dRRfull = dRR; %zeros(size(F));
%dRRfull(find(oIsGood),:) = dRR;
dOOfull = dOO; %zeros(size(F));
%dOOfull(find(oIsGood),:) = dOO; %(O-Oexp)./Oexp; %
%dIIfull = dII;

%goodIds = getGoodIds(matfileRaw,matfilePost);
goodIds = find(matfilePost.goodIds);

% 
% minIsGood = min(dOOfull,[],2)<minTh;
% maxIsGood = max(dOOfull,[],2)>maxTh;
% magIsGood = mean(dOOfull,2)<magTh;
% 
% goodIds = find(rsqIsGood.*ampIsGood.*oIsGood.*minIsGood.*maxIsGood.*magIsGood.*rgccIsGood);

oData = dOOfull(goodIds,:); %Fsc(goodIds,:) - F0full(goodIds,:);
yData = dYYfull(goodIds,:);
rData = dRRfull(goodIds,:);
% iData = dIIfull(goodIds,:);

% isGreenDominant = zeros(size(goodIds));
% for j=1:length(isGreenDominant)
%     gtmp = corrcoef(oData(j,:),yData(j,:));
%     rtmp = corrcoef(oData(j,:),rData(j,:));
%     if abs(gtmp(2))>abs(rtmp(2)); isGreenDominant(j) = 1; end
% end
% oData = oData(find(isGreenDominant),:); %Fsc(goodIds,:) - F0full(goodIds,:);
% yData = yData(find(isGreenDominant),:);
% rData = rData(find(isGreenDominant),:);

% % pca pre-processing
% [coeff,score,latent] = pca(data);
% nCut = 100; %50;
% d = coeff(:,1+(1:nCut)); %[mvnrnd(Mu1,Sigma1,N*p(1)) ; mvnrnd(Mu2,Sigma2,N*p(2)); mvnrnd(Mu3,Sigma3,N*p(3))];
d = oData; 

% % kmeans
% K = 20; %20;%50;
% [~,~,idx]=kmeans(d,K);
% %idx = randi(K,[1,size(d,1)]);
% if length(idx)==1; idx = 1:size(d,1); K=size(d,1); end
% 
% m = zeros(K,size(d,2));
% for k=1:K
%     m(k,:)=mean(d(idx==k,:));
% end

% % hierarchical clustering
%indTemp = zeros(size(F,1),1);
%indTemp(find(oIsGood),:) = clustInd; %(O-Oexp)./Oexp; %
%clustIdxGood = indTemp(goodIds);
idx = clustInd; %(find(isGreenDominant)); %clustInd
K=length(unique(idx)); %size(clustData,1);
m=clustData;





% version based on max dev
m(isnan(m))=0;
maxdevs = movmad( smoothdata(m,2,'movmean',20), 100,2);
% [~,pOrd] = sort( max(maxdevs,[],2), 'descend');
% pOrdFull = zeros(size(idx));
% ctr=0;
% for k=1:K
%     i = find(idx==pOrd(k));
%     pOrdFull(ctr+(1:length(i))) = i;
%     ctr = ctr+length(i);
% end

% version seeded on max dev, then corr
[~,seed] = max( max(maxdevs,[],2));
mc = corrcoef(m');
%[~,pOrd] = sort(mc(seed,:),'descend');
pOrd = zeros(1,K);
pOrd(1)=seed;
for k=1:K-1
    tmp = nanmean(mc(pOrd(1:k),:),1);
    tmp(pOrd(1:k))=nan;
    [~,loc] = nanmax(tmp);
    pOrd(k+1)=loc;
end
pOrdFull = zeros(size(idx));
ctr=0;
for k=1:K
    i = find(idx==pOrd(k));
    pOrdFull(ctr+(1:length(i))) = i;
    ctr = ctr+length(i);
end


% % version sorting on corr with behavior
% c = corr(data',alignedBehavior.legVar');
% ctmp = zeros(1,K);
% for k=1:K
%     ctmp(k) = max(c(idx==k));
% end
% [~,pOrd] = sort(ctmp,'descend');
% pOrdFull = zeros(size(idx));
% ctr=0;
% for k=1:K
%     i = find(idx==pOrd(k));
%     pOrdFull(ctr+(1:length(i))) = i;
%     ctr = ctr+length(i);
% end




% figure (raw)  ------------------------------------------------------------------

for flist = 1:3
    if flist==1
        pData = oData;
        dFlag = ' \Delta O/O';
        fnm = 'dOO';
    elseif flist==2
        pData = yData;
        dFlag = ' \Delta Y/Y';
        fnm = 'dYY';
    elseif flist==4
        pData = iData;
        dFlag = ' \Delta I/I';
        fnm = 'dII';
    else
        pData = rData;
        dFlag = ' \Delta R/R';
        fnm = 'dRR';
    end
    
    % generate xticks and labels
    tks = floor(time(1)/60):floor(time(end)/60);
    tkLoc = zeros(size(tks));
    tkLab = cell(size(tks));
    for j=1:length(tks)
        if mod(tks(j),2); tkLab{j}='';
        else; tkLab{j}=num2str(j);
        end
        [~,tkLoc(j)] = min(abs(time-60*tks(j)));
    end
    
    f2 = figure;
    set(gcf,'color','w')
    pos=get(gcf,'Position');
    pos(3)=500;%pos(3)*1.5;
    pos(4)=800;%pos(4)*1.5;
    set(gcf,'Position',pos)
    f2.InvertHardcopy = 'off';
    f2.PaperUnits = 'points';
    f2.PaperSize = 1.1*[pos(3) pos(4)];
    
    subplot(20,1,3:10); imagesc(pData(pOrdFull,:)); xlim([1 size(pData,2)]); caxis([0.1 0.9])
    ylabel('Cell Number','fontsize',12); xlabel('Time (min)','fontsize',12);
    set(gca,'XTick',tkLoc); set(gca,'XTickLabel',tkLab);
    colormap hot
    setFigColors;
    tmpA=get(gca,'Position');
    
    subplot(20,1,2); plot(behNorm,'color',[.9,.9,.9],'linewidth',1); xlim([1 size(pData,2)]); ylim([min(behNorm) max(behNorm)]); axis off
    tmpB=get(gca,'Position');
    set(gca,'Position',[tmpB(1),tmpB(2)-.4*(tmpB(2)-tmpA(2)-tmpA(4)),tmpB(3),tmpB(4)+.4*(tmpB(2)-tmpA(2)-tmpA(4))]);
    
    subplot(20,1,1); plot(alignedBehavior.stim,'linewidth',1); hold all; plot(alignedBehavior.drink,'linewidth',1); xlim([1 size(pData,2)]); ylim([0 1]); axis off
    tmpC=get(gca,'Position');
    set(gca,'Position',[tmpC(1),tmpC(2)-.4*(tmpC(2)-tmpB(2)-tmpB(4)),tmpC(3),tmpC(4)+.4*(tmpC(2)-tmpB(2)-tmpB(4))]);
    title([expID,dFlag,ccFlag],'color',[.9,.9,.9]);%,'interpreter','none')
    
    % make map of footprints --------------------------------------------------
    bkTh = 0.1;
    groupColors = flipud(jet(K));
    R = zeros(d1,d2);
    G = zeros(d1,d2);
    B = zeros(d1,d2);
    Agood = A(:,goodIds);
    
    for k=1:K
        kIds = idx==pOrd(k); %goodIds(idx==pOrd(k));
        Ar = reshape( max(full(Agood(:,kIds)),[],2), d1,d2,d3);
        Ar1 = max(Ar,[],3);
        R = R + Ar1*groupColors(k,1);
        G = G + Ar1*groupColors(k,2);
        B = B + Ar1*groupColors(k,3);
    end
    mx = max([max(R(:)),max(G(:)),max(B(:))]);
    R = R/mx; G = G/mx; B = B/mx;
    RGB = cat(3,R,G,B);
    
    
    
    % make colorbar to relate footprints to traces ----------------------------
    atmp = tmpA;%get(gca,'position');
    axes('position',[atmp(1)+atmp(3)+.015,atmp(2),.02,atmp(4)])
    [~,idl] = sort(idx);
    itmp = groupColors(idx,:);
    itmp = permute(repmat(itmp,1,1,4),[1,3,2]);
    colorIDs = itmp(idl,:,:);
    whiteStrips = zeros(size(colorIDs));
    whiteStrips(1,3:4,:)=1;
    stripEnd=true;
    for j=2:size(whiteStrips,1) % this is a dumb way to do this
        if (colorIDs(j-1,1,1)==colorIDs(j,1,1))&&(colorIDs(j-1,1,2)==colorIDs(j,1,2))&&(colorIDs(j-1,1,3)==colorIDs(j,1,3))
            whiteStrips(j,:,:)=whiteStrips(j-1,:,:);
        else
            stripEnd=~stripEnd;
            if stripEnd; whiteStrips(j,3:4,:)=1;
            else; whiteStrips(j,1:2,:)=1;
            end
        end
    end
    colorIDs(find(whiteStrips))=0;
    imagesc(colorIDs)
    setFigColors;
    axis off
    
    
    % show map of footprints --------------------------------------------------
    subplot(20,1,12:20);
    bk1 = max(R+G+B,[],3); bk1 = bk1/quantile(bk1(:),0.95); bk1(bk1>1)=1;
    al = bk1>bkTh; %squeeze(bk1.*bk.*(bk2>bkTh)); %
    b = squeeze(max(Ysum,[],3));
    b = b/max(b(:));
    b = cat(3,b,b,b);
    Kadj = imadjust(b, [0 1], [0 1], 0.6 );
    imagesc(Kadj)
    rgb_adj = imadjust(RGB, [0.1 0.7], [0 1], 0.6 );
    hold on; im=imagesc(rgb_adj);%/q);
    set(im, 'AlphaData', 4*al);%4*al);
    axis off;
    %annotation('line',.905+[0 25*ypx/params.yum],[.05 .05],'linewidth',2,'color',[.6 .6 .6])
    %annotation('textbox',[.905 .047 25*ypx/params.yum .01],'String','25um','color',[.5 .5 .5],'fontsize',12,'linestyle','none','HorizontalAlignment','center')
    setFigColors;
    f2.InvertHardcopy = 'off';
    if ~isfolder([expDir,'_plots/',fnm]); mkdir([expDir,'_plots/',fnm]); end
    saveas(f2, [expDir,'_plots/',fnm,'/',expID,'_',fnm,'_from',ccFlag(end-2:end),'.png'])

end

function setFigColors
set(gca,'Fontsize',14)
set(gca,'color','none');%'k')
set(gcf,'color','none');%'k');
set(gca,'xcolor',[.9 .9 .9]);
set(gca,'ycolor',[.9 .9 .9]);
