
function plotAndMapCuratedComponents

% supply a list of indices and plot the traces and show footprint for those
% cells.

addpath(genpath( '~/Dropbox/_code/'))
expDir = '/Users/evan/Dropbox/_sandbox/sourceExtraction/good/feeding/';%_runAndFeed/'; %
expID = '190110'; %'190424_f3'; %'190218Crz'; %'0327_f3'; %'0824_f2r2_cur'; %'0824_f3r1'; %'0221'; %'0327_f3'; %'0321'; %

fromGreenCC = true; %false;
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
matfileBeh = matfile([expDir,expID,'/alignedBehavAndStim.mat']);
infoFile = dir([expDir,expID,'/info/*.mat']);%'/Volumes/dataFast/habaRegistered/2018_08_24_odor/mats/new/fly3run2/'; %'/Users/evan/Desktop/hungerRaw/';%'/Volumes/SCAPEdata1/scratchData/2018_08_01_IRtest/matfiles/registered/';%'/Volumes/data/_scape/data/_outMats/'; %
if length(infoFile)>1; infoFile=infoFile(end); end
load([expDir,expID,'/info/',infoFile.name]);
sz = size(Ysum);

% -------------------------------------------------------------------------
% toggle findComponents to look at which components to plot, or to make fig
findComponents = false; %true;
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------



% ------------ DIMENSIONS -----------------------------
try
    params.xum = sz(1)*info.GUIcalFactors.x_umPerPix;
catch
    %warning('no x_um field available')
    params.xum = sz(1)*info.GUIcalFactors.xK_umPerVolt*info.daq.scanAngle/(info.daq.pixelsPerLine-1);
end
params.yum = sz(2)*info.GUIcalFactors.y_umPerPix;
params.zum = sz(3)*info.GUIcalFactors.z_umPerPix;
totWidth = 1.1*params.yum; %(params.zum + params.yum);
params.zpx = params.zum/totWidth;
params.ypx = params.yum/totWidth;
params.xpx = params.xum/totWidth;



% ------------ MAKE FINDER PLOT OR FINAL PLOT -----------------------------
if findComponents
    Ar = reshape(full(sum(A,2)),sz(1),sz(2),sz(3));
    c = 1:size(A,2);
    Am = A*sparse(diag(c));
    Ad = reshape(full(max(Am,[],2)),sz(1),sz(2),sz(3));
    figure; imagesc(max(Ad,[],3));
    figure; imagesc(max(Ysum,[],3));
else
    %ids = [6,18,34]; %190424_f3 one side from R
    %ids = [19,12,22,33]; %190418 one side from R
    %ids = [8,13,227,282]; %0110 Crz cells from Y
    ids = [8,13,16,14]; %0110 Crz one side from Y
    %ids = [84,62,261,245];%0312f4 Crz cells from Y
    %ids = [59,36,206,230];%0312f4 Crz cells from R
    %ids = [78,52,346,408];%0312f4 Crz complement from R
    figName = [expID,'_',num2str(ids)];
    figName(strfind(figName,' '))='_';
    makeFigs(params,sz,A,dOO,matfileBeh,ids,Ysum,expDir,figName)
end





% --------------------- extra functions  ----------------------------------
% -------------------------------------------------------------------------

function makeFigs(params,sz,A,dOO,matfileBeh,goodIds,Ysum,expDir,figName)
bkTh = 0.1;
Agood = A(:,goodIds);
dOOgood = dOO(goodIds,:);
colors = hsv(length(goodIds));


Ar1 = reshape( sum(full(Agood*sparse(diag(colors(:,1)'))),2), sz(1),sz(2),sz(3));
Ar2 = reshape( sum(full(Agood*sparse(diag(colors(:,2)'))),2), sz(1),sz(2),sz(3));
Ar3 = reshape( sum(full(Agood*sparse(diag(colors(:,3)'))),2), sz(1),sz(2),sz(3));

R = max(Ar1,[],3);%/max(Ar1(:));
G = max(Ar2,[],3);%/max(Ar2(:));
B = max(Ar3,[],3);%/max(Ar3(:));


mx = max([max(R(:)),max(G(:)),max(B(:))]);
R = R/mx; G = G/mx; B = B/mx;
RGB = cat(3,R,G,B);
bk1 = max(R+G+B,[],3); bk1 = bk1/quantile(bk1(:),0.95); bk1(bk1>1)=1;
al = bk1>bkTh; %squeeze(bk1.*bk.*(bk2>bkTh)); %
b = squeeze(max(Ysum,[],3));
b = b/max(b(:));
b = cat(3,b,b,b);
Kadj = imadjust(b, [0 1], [0 1], 0.6 );

pi.totScale = 0.8; % ratio of height to width for the whole figure. image dims are compensated to keep correct scaling
figw = 500;
pi.figpos = [368 286 figw figw*pi.totScale]; % figure position
f2 = figure('position', pi.figpos);
axes('position',[.05  .05/pi.totScale  params.ypx  params.xpx/pi.totScale]); %subplot(1, 3, 1);
annotation('line',.875+[0 25*params.ypx/params.yum],[.11 .11],'linewidth',2,'color',[.6 .6 .6])
annotation('textbox',[.875 .1 25*params.ypx/params.yum .01],'String','25um','color',[.5 .5 .5],'fontsize',12,'linestyle','none','HorizontalAlignment','center')

imagesc(Kadj)
rgb_adj = imadjust(RGB, [0.1 0.7], [0 1], 0.6 );
hold on; im=imagesc(rgb_adj);%/q);
set(im, 'AlphaData', 4*al);%4*al);
axis off;
title(figName,'interpreter','none')
%annotation('line',.905+[0 25*params.ypx/params.yum],[.05 .05],'linewidth',2,'color',[.6 .6 .6])
%annotation('textbox',[.905 .047 25*params.ypx/params.yum .01],'String','25um','color',[.5 .5 .5],'fontsize',12,'linestyle','none','HorizontalAlignment','center')
f2.InvertHardcopy = 'off';
if ~isfolder([expDir(1:end-5),'_plots/_summary/']); mkdir([expDir(1:end-5),'_plots/_summary/']); end
saveas(f2, [expDir(1:end-5),'_plots/_summary/',figName,'.png'])
makeTraceFig(matfileBeh, dOOgood, colors, pi.figpos, expDir, [figName,'_traces'])


% figure; plot(matfileBeh.time,O(8,:),'linewidth',1,'color',[1,.8,0]); box off
% hold all; plot(matfileBeh.time,Oexp(8,:),'linewidth',1,'color',[.8,.8,.8])
% xlim([50 550])
% ylabel('Fluorescence Ratio (G/R)','fontsize',12); xlabel('Time (s)','fontsize',12);
% set(gca,'Fontsize',14)
% set(gca,'color','none');set(gcf,'color','none');%'k');
% set(gca,'xcolor',[.9 .9 .9]); set(gca,'ycolor',[.9 .9 .9]);


function makeTraceFig(matfileBeh, dataForIm, colors, figSize, expDir, figtitle)

f2 = figure;
set(gcf,'color','w')
pos=get(gcf,'Position');
pos(3)=figSize(3);%pos(3)*1.5;
pos(4)=.5*figSize(4);%pos(4)*1.5;
set(gcf,'Position',pos)
f2.InvertHardcopy = 'off';
f2.PaperUnits = 'points';
f2.PaperSize = 1.1*[pos(3) pos(4)];

time = matfileBeh.time;
beh = matfileBeh.alignedBehavior;
area(time,beh.drink*max(dataForIm(:)),'FaceColor',[.4,.4,.4]); hold on

for j=1:size(dataForIm,1)
    plot(time, dataForIm(j,:),'linewidth',1,'color',colors(j,:)); hold on;
end
xl = [10*floor(time(1)/10), 10*ceil(time(end)/10)];
xlim(xl); box off; set(gca,'XTick',100:100:xl(2)); 
xlabel('Time (s)','fontsize',14);
ylabel('\Delta O/O','fontsize',14);
title(figtitle,'color',[.9,.9,.9])
setFigColors;
f2.InvertHardcopy = 'off';
f2.PaperPositionMode = 'auto';
saveas(f2, [expDir(1:end-5),'_plots/_summary/',figtitle,'.png'])



function setFigColors
set(gca,'Fontsize',14)
set(gca,'color','none');%'k')
set(gcf,'color','none');%'k');
set(gca,'xcolor',[.9 .9 .9]);
set(gca,'ycolor',[.9 .9 .9]);
