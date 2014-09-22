% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.
% 
% TLD is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% TLD is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with TLD.  If not, see <http://www.gnu.org/licenses/>.

function [bb,conf] = tldExample(opt)

global tld; % holds results and temporal variables

% INITIALIZATION ----------------------------------------------------------

opt.source = tldInitSource(opt.source); % select data source, camera/directory

figure(2); set(2,'KeyPressFcn', @handleKey); % open figure for display of results
finish = 0; function handleKey(~,~), finish = 1; end % by pressing any key, the process will exit

while 1
    source = tldInitFirstFrame(tld,opt.source,opt.model.min_win); % get initial bounding box, return 'empty' if bounding box is too small
    if ~isempty(source), opt.source = source; break; end % check size
end

opt.source.bb = [nan;nan;nan;nan];
if (~isnan(opt.source.bb(1)))
	tld = tldInit(opt,[]); % train initial detector and initialize the 'tld' structure
	tld = tldReInit(opt,tld,1);
	tld = tldDisplay(0,tld); % initialize display
else
	tld = tldInit(opt,[]);
	tld.handle = imshow(tld.img{1}.input,'initialmagnification','fit');
    bb_draw(tld.source.bb);
end

% RUN-TIME ----------------------------------------------------------------

for i = 2:length(tld.source.idx) % for every frame
    
	if mod(i, tld.update_freq) == 0
		%% Every tld.update_freq frame, initialize tracker with detection result from R-CNN
		tld.bb(:, i) = [126; 56; 188; 158];
		I = tld.source.idx(i); % get current index
		tld.img{I} = img_get(tld.source,I); % grab frame from camera / load image
		continue;
	end
	
	%if isempty(tld.bb(:, i-1))
	%	tld.bb{i} = [];
	%end
	
	if ~isnan(tld.bb(1, i-1))
		tld = tldReInit(opt,tld,i-1);
		tld = tldProcessFrame(tld,i); % process frame i
		tldDisplay(1,tld,i); % display results on frame i
	end
    
    if finish % finish if any key was pressed
        if tld.source.camera
            stoppreview(tld.source.vid);
            closepreview(tld.source.vid);
             close(1);
        end
        close(2);
        bb = tld.bb; conf = tld.conf; % return results
        return;
    end
    
    if tld.plot.save == 1
        img = getframe;
        imwrite(img.cdata,[tld.output num2str(i,'%05d') '.png']);
    end
        
    
end

bb = tld.bb; conf = tld.conf; % return results

end
