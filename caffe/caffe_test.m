addpath('rp/matlab/');
addpath('rp/cmex/');

data_path = '../data/';

configFile = 'rp/config/rp_4segs.mat'; 
configParams = LoadConfigFile(configFile);

use_gpu = true;
input_batch_size = 250;
model_def_file = './vocnet_deploy.prototxt';
model_file = '/mnt/neocortex/scratch/tsechiw/caffe/build/caffe_intunet_train_iter_140000';

caffe('init', model_def_file, model_file);

if exist('use_gpu', 'var') && use_gpu
  caffe('set_mode_gpu');
else
  caffe('set_mode_cpu');
end

% put into test mode
caffe('set_phase_test');

d = dir([data_path '*.jpg']);

t=0;
for i = 1:size(d,1)
	tic;
	im = imread([data_path d(i).name]);
	proposals = RP(im, configParams);
	scores_matrix = detect_cnn(im, proposals, 100, input_batch_size, 5);
	t = t + toc;
	epi = scores_matrix(2,:);
	voc = scores_matrix(3,:);
	tra = scores_matrix(4,:);
	car = scores_matrix(5,:);
	[scores, region] = max(scores_matrix, [], 2);
	scores
	clf;
	imshow(im);
	hold on;
	for j = 2:length(region)
		if scores(j) >= .85
			idx = region(j);
			bbox = [proposals(idx, 1), proposals(idx, 2), proposals(idx, 3)-proposals(idx, 1), proposals(idx, 4)-proposals(idx, 2)];
			rectangle('position', bbox, 'edgecolor', 'g', 'linewidth', 2);
		end
	end
	drawnow;
	fprintf('Processed %d/%d frames\n', i, size(d,1));
end

fprintf('Processing speed = %.3f\n', t/size(d,1));

%end
%{
imshow(im);
hold on;
for i = 2:length(region)
	if score(i) >= .85
		idx = region(i);
		bbox = [proposals(idx, 1), proposals(idx, 2), proposals(idx, 3)-proposals(idx, 1), proposals(idx, 4)-proposals(idx, 2)];
		rectangle('position', bbox, 'edgecolor', 'g', 'linewidth', 2);
	end
end
%}