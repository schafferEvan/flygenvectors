function vid = plot4Dproj_ES(Y, maskY, sizY, params, stimOn, A, C, b, f)
% plot4Dproj(Y, maskY, sizY, params)
% Displays XY, YZ and XZ projections of Y over time.
% inputs:
% Y : 3D or 4D matrix (2DxT, 3D, 3DxT)
% maskY: mask to overlay on Y (as a contour, for values greater then 0)
% sizY: spatial dimensions of Y (2D or 3D);
% params: extra parameters that edit the figure fitures
% Notes:
% mask must have the same 'spatial dimensions' as Y
% the extra dimention will be read as a different masks. so far the
% code only works for cases where last dimention of maskY is 1 or matches
% the last dimention of Y.

% update: 01/26/2018 - now can produce movie of CNMF-parsed image

global pi figH
pi = [];
pi.iter = false; % update caxis iteratively (every timepoint)
pi.range = []; % range to use by caxis
pi.cbgate = false; % display colorbar
pi.lag = 0.01; % lag between iterations (or timepoints)
pi.maskcor = [.9 .9 .9]; % default color of mask contour
pi.maskout = 0; % gate to zero all the Y values where maskY == 0
pi.totScale = 0.8; % ratio of height to width for the whole figure. image dims are compensated to keep correct scaling
pi.figpos = [368 286 800 800*pi.totScale]; % figure position

%pi.cormap = 'jet'; % colormap of background image


%fop = fields(pi);
fop2 = fields(params);
if exist('params', 'var') && ~isempty(params)
    %     for ii = 1:length(fop)
    %         if isfield(params, fop{ii}) && ~isempty(params(1). (fop{ii}))
    %             pi.(fop{ii}) = params(1).(fop{ii});
    %         end
    %     end
    for ii = 1:length(fop2)
        pi.(fop2{ii}) = params.(fop2{ii});
    end
end

if ~isfield(pi,'acqRate')
    %pi.frameRate = 50;
    pi.acqRate = 10;
end

% flatten all
% Y = reshape(full(Y), prod(sizY), []);
% maskY = reshape(full(maskY), prod(sizY), []);
% 
% % reshape Y and maskY if it is flatten
% Y = reshape(full(Y), [sizY, size(Y, 2)]);
% maskY = reshape(full(maskY), [sizY, size(maskY, 2)]);

Y = double(Y);

if size(Y,4)>1
    %Y = Y/params.maxY; % if imaging Y --- THIS IS NOW HANDLED OUTSIDE
else
    Y = Y/max(max(max(Y))); %*full(params.maxY) % if Y is a static background
end
Y(Y>1)=1;

%% Plotting
figH = figure('position', pi.figpos);
figH.Color = [.1 .1 .1];
%colormap(pi.cormap)

if pi.maskout && size(maskY, 4) == 1
   Y = bsxfun(@times, Y, double(maskY > 0));
end

% if isempty(pi.range) && pi.iter == 0
%    pi.range = [100 params.maxY]; %[min(Y(:)) max(Y(:))];
% end
pi.range = [0 1];

if nargin<6
    A = [];
    C = []; 
    b = [];
    f = [];
end

%convert to indices to estimate Z mean and calculate index correspondence
%in flattened image.                                                             
%indexing might be right, but hasn't been
%checked. need to feed indices into structure below and calculate mean on
%3rd dimension 


cnmfx = struct('A',A,'C',C,'b',b,'f',f);
if ~isempty(A)
    %[Ax1,Ax2,Ax3] = ind2sub([params.options.d1,params.options.d2,params.options.d3],A);
    %cnmfx.Ax1 = Ax1;
    %cnmfx.Ax2 = Ax2;
    %cnmfx.Ax3 = Ax3;
    % the above is wrong. the right way to do this is to use find to get
    % the indices corresponding to nonzero values of A for each ROI, and
    % work with that.
    cnmfx.br = reshape(cnmfx.b,params.options.d1,params.options.d2,params.options.d3);
    cnmfx.imMax = -inf;
    cnmfx.imMin = inf;
    if isfield(params,'maxY')
        cnmfx.imMax = params.maxY;
        cnmfx.imMin = params.minY;
    else
        for j=1:size(A,1)
            cnmfx.imMax = max( max(max(A(j,:)*C(:, round( size(C,2)/2 ):end ))), cnmfx.imMax);
            cnmfx.imMin = min( min(min(A(j,:)*C)), cnmfx.imMin);
        end
    end
    pi.range = [cnmfx.imMin cnmfx.imMax];
end
clear A C b f

vid = plotmultiproj(Y, maskY, sizY, stimOn, cnmfx, params);
end




function vid = plotmultiproj(Y, maskY, sizY, stimOn, cnmfx, params)
global pi figH
pi.d2proj = [3 2 1];
pi.axesname = {'', ''; 'Z', 'Y'; 'X', 'Z'};
pi.c2plot = [2, 1; 3, 1; 3, 2];
% if isempty(pi.range)
%     pi.range = [min(Y(:)) max(Y(:))];
% end

totWidth = 1.1*(params.zum + params.yum);
zpx = params.zum/totWidth;
ypx = params.yum/totWidth;
xpx = params.xum/totWidth;

% hAxes(1) = axes('position',[.325  .3  .6  .6]); %subplot(1, 3, 1);
% hAxes(2) = axes('position',[.075  .3  .22  .6]); %subplot(1, 3, 2);
% hAxes(3) = axes('position',[.325  .05  .6  .22]); %subplot(1, 3, 3);
% hAxes(4) = axes('position',[.075  .05  .22  .22]); %subplot(1, 3, 3);
hAxes(1) = axes('position',[.05+zpx  (.05+zpx)/pi.totScale  ypx  xpx/pi.totScale]); %subplot(1, 3, 1);
hAxes(2) = axes('position',[.04      (.05+zpx)/pi.totScale  zpx  xpx/pi.totScale]); %subplot(1, 3, 2);
hAxes(3) = axes('position',[.05+zpx  .04/pi.totScale      ypx  zpx/pi.totScale]); %subplot(1, 3, 3);
hAxes(4) = axes('position',[.04      .04/pi.totScale      zpx  zpx/pi.totScale]); %subplot(1, 3, 3);

annotation('line',.905+[0 25*ypx/params.yum],[.05 .05],'linewidth',2,'color',[.6 .6 .6])
annotation('textbox',[.905 .047 25*ypx/params.yum .01],'String','25um','color',[.5 .5 .5],'fontsize',12,'linestyle','none','HorizontalAlignment','center')


fprintf(['Plotting ', num2str(size(Y, 4)), ' volume(s)\n'])
if isfield(pi,'savePath') && isfield(pi,'trialName')
    if isfield(params,'vid')
        vid = params.vid;
    else
        vid = VideoWriter([pi.savePath,pi.trialName]);
        vid.FrameRate = 30;
        open(vid)
    end
end
if isfield(cnmfx,'C')
    if ~isempty(cnmfx.C)
        T = size(cnmfx.C,2);
    else
        T = size(Y, 4);
    end
else
    T = size(Y, 4);
end
    
for t = 1:T %size(Y, 4)
    for a_idx = 1:3
        %% plot Y
            if isempty(cnmfx.A)
                plotProjZColor(hAxes, squeeze(Y(:, :, :, t)), a_idx, t);
            else
                cnmfx.br = Y;
                plotProjCnmfColor(hAxes, cnmfx, a_idx, t);
                %baseline = cnmfx.br*cnmfx.f(t);
                %plotProjCnmfResidual(hAxes, squeeze(Y(:, :, :, t)), cnmfFrame, a_idx, t);
            end
            %plotProjZColor(hAxes, squeeze(permute(Y(:, :, :, t),[2,1,3,4])), a_idx, t);
        
        %% overlay contour maskY
        if ~isempty(maskY)
            if size(Y, 4) == size(maskY, 4) % if mask is same size uses each for each Y(:, :, :, t)
                plotoverlay(hAxes, a_idx, maskY(:, :, :, t))
            elseif size(maskY, 4) == 1 % otherwise it uses the same for all (1)
                plotoverlay(hAxes, a_idx, maskY(:, :, :))
            end
        end
    end
    axes(hAxes(4)); axis off
    xlim([0 1]); ylim([0 1])
    if length(stimOn)==2
        isStim = ( (t>=stimOn(1)) && (t<=stimOn(2)) );
    else
        isStim = stimOn(t+pi.kTot);
    end
    if isStim
        rectangle('Position',[0.1 .1 .8 .8],'Facecolor',isStim*[.8 .8 .8]);
    else
        rectangle('Position',[0.1 .1 .8 .8],'Facecolor',figH.Color);
    end
    drawnow; %pause(pi.lag)
    if isfield(pi,'savePath') && isfield(pi,'trialName')
        writeVideo(vid, getframe(figH))
    end
end
if isfield(pi,'savePath') && isfield(pi,'trialName') && ~params.concatenate
    close(vid)
end
end




function plotProjCnmfColor(hAxes, cnmfx, hi, ti)
global pi

% reconstruct background as b*f(t)
% make this flattened image and populate color dimension as gray
% (loop over) ROI dimension of A, for each ROI, set indexed values equal to <Z>*C(i,t) 

%[v,l] = max(cnmfx.br(:,:,2:end-1)*cnmfx.f(ti), [], pi.d2proj(hi));
if isfield(pi,'cropy')
    [v,l] = max(cnmfx.br(1:pi.cropx,1:pi.cropy,2:end-1), [], pi.d2proj(hi));
else
    [v,l] = max(cnmfx.br(:,:,2:end-1), [], pi.d2proj(hi));
end
v = squeeze(v); l = squeeze(l);
imhsv = cat(3,zeros(size(v,1),size(v,2)),zeros(size(v,1),size(v,2)),v/max(max(v)));
rois = reshape(cnmfx.A*cnmfx.C(:,ti),size(cnmfx.br,1),size(cnmfx.br,2),size(cnmfx.br,3));
if isfield(pi,'cropy'); rois = rois(:,1:pi.cropy,:); end
if isfield(pi,'cropx'); rois = rois(1:pi.cropx,:,:); end
[rv,rl] = max(rois, [], pi.d2proj(hi));
rv = squeeze(rv/cnmfx.imMax); rl = squeeze(rl);
%rv = squeeze(rv); rl = squeeze(rl);

% if hi==1
%     roihsv = cat(3, rl/size(cnmfx.br,pi.d2proj(hi)), ones(size(rl)), rv);
% elseif hi==2
%     roihsv = cat(3, repmat(1:size(rl,2),size(rl,1),1)/size(rl,2), ones(size(rl)), rv);
% else
%     rv = rv'; rl=rl';
%     roihsv = cat(3, repmat((1:size(rl,1))',1,size(rl,2))/size(rl,1), ones(size(rl)), rv);
%     imhsv = permute(imhsv,[2,1,3]);
% end
if hi==1
    tmp = -.25+.99*rl/size(cnmfx.br,pi.d2proj(hi)); tmp = tmp + (tmp<0); % add 1 if tmp<0
    roihsv = cat(3, tmp, ones(size(rl)), rv);
elseif hi==2
    tmp = -.25+.99*repmat(1:size(rl,2),size(rl,1),1)/size(rl,2); tmp = tmp + (tmp<0); % add 1 if tmp<0
    roihsv = cat(3, tmp, ones(size(rl)), rv);
else
    rv = rv'; rl=rl';
    tmp = -.25+.99*repmat((1:size(rl,1))',1,size(rl,2))/size(rl,1); tmp = tmp + (tmp<0); % add 1 if tmp<0
    roihsv = cat(3, tmp, ones(size(rl)), rv);
    imhsv = permute(imhsv,[2,1,3]);
end

axes(hAxes(hi))
%imagesc(hsv2rgb(imhsv), 'Parent', hAxes(hi)); hold on
imrgb = imadjust( hsv2rgb(imhsv), [0 1], [0 1], 0.7 ); % last argument (gamma) amplifies dark areas
imagesc(imrgb); hold on
imrgb = imadjust( hsv2rgb(roihsv), [0 1], [0 1], 0.7 ); % last argument (gamma) amplifies dark areas
im = imagesc(imrgb); hold off
set(im, 'AlphaData', 4*rv);%/(.5*cnmfx.imMax));

if pi.iter; pi.range = [min(cnmfx.br(:)) max(cnmfx.br(:))]; end % update caxis iteratively at each t
if ~isempty(pi.range); caxis(hAxes(hi), pi.range); end
if pi.cbgate; if hi==1; colorbar(hAxes(hi)); end; end
xlabel(hAxes(hi), pi.axesname{hi, 1},'color',[.7 .7 .7]); 
ylabel(hAxes(hi), pi.axesname{hi, 2},'color',[.7 .7 .7]);
totTitle = ['time = ',num2str(pi.Ttot+.1*round(10*ti/pi.acqRate)),' (sec)'];
if isfield(pi,'trialName')
    totTitle = {pi.trialName; totTitle};
end
if hi==1; title(hAxes(hi), totTitle,'color',[.7 .7 .7],'Interpreter','none'); end
set(hAxes(hi), 'YTick', []); set(hAxes(hi), 'XTick', [])
end




function plotProjZColor(hAxes, Im, hi, ti)
global pi
[v,l] = max(Im, [], pi.d2proj(hi));
v = squeeze(v); l = squeeze(l);
vscaled = v; %(v-min(v(:)))/(max(v(:))-min(v(:))); % DON'T RESCALE!
if hi==1 
    tmp = -.25+.99*l/size(Im,pi.d2proj(hi)); tmp = tmp + (tmp<0); % add 1 if tmp<0
    imhsv = cat(3, tmp, ones(size(l)), vscaled);
elseif hi==2
    tmp = -.25+.99*repmat(1:size(l,2),size(l,1),1)/size(l,2); tmp = tmp + (tmp<0); % add 1 if tmp<0
    imhsv = cat(3, tmp, ones(size(l)), vscaled);
else
    vscaled = vscaled'; l=l';
    tmp = -.25+.99*repmat((1:size(l,1))',1,size(l,2))/size(l,1); tmp = tmp + (tmp<0); % add 1 if tmp<0
    imhsv = cat(3, tmp, ones(size(l)), vscaled);
end
imrgb = imadjust( hsv2rgb(imhsv), pi.range, pi.range, 0.7 ); % last argument (gamma) amplifies dark areas
imagesc(imrgb, 'Parent', hAxes(hi))
%if pi.iter; pi.range = [100 params.maxY]; end % update caxis iteratively at each t
%if ~isempty(pi.range); caxis(hAxes(hi), pi.range); end % caxis(hAxes(hi), pi.range);
%if pi.cbgate; if hi==1; colorbar(hAxes(hi)); end; end
%caxis(hAxes(hi), pi.range)
xlabel(hAxes(hi), pi.axesname{hi, 1},'color',[.7 .7 .7]); 
ylabel(hAxes(hi), pi.axesname{hi, 2},'color',[.7 .7 .7]);
totTitle = ['time = ',num2str(pi.Ttot+.1*round(10*ti/pi.acqRate)),' (sec)'];
if isfield(pi,'trialName')
    totTitle = {pi.trialName; totTitle};
end
if hi==1; title(hAxes(hi), totTitle,'color',[.7 .7 .7],'Interpreter','none'); end
set(hAxes(hi), 'YTick', []); set(hAxes(hi), 'XTick', [])
end

% function plotproj(hAxes, Im, hi, ti)
% global pi
% imagesc(squeeze(max(Im, [], pi.d2proj(hi))), 'Parent', hAxes(hi))
% if pi.iter; pi.range = [min(Im(:)) max(Im(:))]; end % update caxis iteratively at each t
% if ~isempty(pi.range); caxis(hAxes(hi), pi.range); end
% if pi.cbgate; if hi==1; colorbar(hAxes(hi)); end; end
% xlabel(hAxes(hi), pi.axesname{hi, 1}); 
% ylabel(hAxes(hi), pi.axesname{hi, 2});
% title(hAxes(hi), ['t = ',num2str(.1*round(10*ti/pi.frameRate)),' (s)'],'color',[.9 .9 .9]);
% set(hAxes(hi), 'YTick', []); set(hAxes(hi), 'XTick', [])
% set(gca,'xcolor',[.9 .9 .9]);
% set(gca,'ycolor',[.9 .9 .9]);
% end




function plotoverlay(hAxes, idx, maskY)
global pi
maskProjected = squeeze(max(maskY, [], pi.d2proj(idx))) > 0;
if idx==3
    maskProjected = maskProjected';
end
hold(hAxes(idx) , 'on')
contour(maskProjected, ...
   1, 'color', pi.maskcor,'linewidth',1, 'Parent', hAxes(idx))
end

