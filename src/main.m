%% Load data (TSDF values)

 %   0   : unknown
 %   0.01: visible
 %   >0  : outside
 %   <0  : inside

data = loadFromDataFile('chair04_cutout.dat');
% data = loadFromDataFile('stool_cutout02.dat');
% data = loadFromDataFile('deskNew02_cutout.dat');


%% Volumen from Tango data
V = data;

% create Volumen ('0.01' value means 'visible area')
value = 0.009;

% keep border and interior of the object
mask = abs(V) < value;
V(~mask) = 0;   % empty
V(V~=0)  = 1;   % occupied

figure(1)
plot_volume(V);


%% TSDF values (animation)
figure(2)

sz = size(data);

cmap_bottom = hot;
cmap_top = parula;

% Need two colormaps to see properly. In Tango data:
%   0   : unknown
%   0.01: visible
%   >0  : outside
%   <0  : inside
j=19;
cmap(1:15,:) = cmap_bottom(35-j:2:64-j,:);
cmap(16:30,:) = cmap_top(20-j:3:64-j,:);
cmap(16,:) = [1 1 1]/10;
cmap(31,:) = [.1 .1 .1]*10;

clf
set(gcf,'color','w');

for i=1:sz(3)
% for i=47  
    % display Volumen
    subplot(1,2,1)
    sz = size(V);
    plot_volume(V); xlabel('x'); ylabel('y'); zlabel('z');

        % display plane
        hold on;
        Z = i;
        surface('XData',[0 sz(1); 0 sz(1)],'YData',[0 0; sz(2) sz(2)],...
                'ZData',[Z Z; Z Z], 'FaceColor','texturemap', 'EdgeColor','none');
        alpha(0.5);
        hold off

    % display TSDF values
    subplot(1,2,2)
    colormap(gca, cmap);
    imagesc(data(:,:,i));
    colorbar
    axis off; axis equal
    title('TSDF values')

    pause(0.1)
end

