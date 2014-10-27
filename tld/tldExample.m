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
	figure;
	tld.handle = imshow(tld.img{1}.input,'initialmagnification','fit');
    bb_draw(tld.source.bb);
end

% CAFFE INITIALIZATION------------------------------------------------

addpath('rp/matlab');
addpath('rp/cmex');
addpath('caffe')

configFile = 'rp/config/rp_4segs.mat'; 
configParams = LoadConfigFile(configFile);

% caffe initialization
use_gpu = true;
input_batch_size = 250;
input_size = 100;
model_def_file = './caffe/vocnet_deploy.prototxt';
model_file = '/mnt/neocortex/scratch/tsechiw/caffe/build/caffe_intunet_train_iter_140000';
% set the ID of GPU being used, e.g., 1
caffe('set_device', 1);
caffe('init', model_def_file, model_file);

if exist('use_gpu', 'var') && use_gpu
  caffe('set_mode_gpu');
else
  caffe('set_mode_cpu');
end

% put into test mode
caffe('set_phase_test');

% RUN-TIME ----------------------------------------------------------------

for i = 2:length(tld.source.idx) % for every frame
    
	if mod(i, tld.update_freq) == 0
		%% Every tld.update_freq frame, initialize tracker with detection result from R-CNN
	
				
		%tld.bb(:, i) = [126; 56; 188; 158];
		I = tld.source.idx(i); % get current index
		
		im = imread(tld.source.files(I).name);
		proposals = RP(im, configParams);
		proposals = checkRegion(proposals, 12, 12);
		scores_matrix = detect_cnn(im, proposals, input_size, input_batch_size, 5);
		[score_epi, idx_epi] = max(scores_matrix(2,:)');
		[score_voc, idx_voc] = max(scores_matrix(3,:)');
		[score_tra, idx_tra] = max(scores_matrix(4,:)');
		[score_car, idx_car] = max(scores_matrix(5,:)');
		
		scores = [score_epi; score_voc; score_tra; score_car];
		idxs = [idx_epi; idx_voc; idx_tra; idx_car];
		[max_score, max_idx] = max(scores);
		bbox = proposals(idxs(max_idx), :);
		tld.img{I} = img_get(tld.source,I); % grab frame from camera / load image
		tld.bb(:,i) = bbox';
		tld = tldReInit(opt,tld,i);
	
	elseif ~isnan(tld.bb(1, i-1))
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
