addpath('rp/matlab');
addpath('rp/cmex');
addpath('caffe')

configFile = 'rp/config/rp_4segs.mat'; 
global configParams;
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