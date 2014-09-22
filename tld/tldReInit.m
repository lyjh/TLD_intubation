function tld = tldReInit(opt,tld,I)

% Scanning grid
[tld.grid tld.scales] = bb_scan(tld.bb(:,I),size(tld.source.im0.input),tld.model.min_win);

% Features
tld.nGrid     = size(tld.grid,2);
tld.features  = tldGenerateFeatures(tld.model.num_trees,tld.model.num_features,0);

% Initialize Detector
fern(0); % cleanup
fern(1,tld.source.im0.input,tld.grid,tld.features,tld.scales); % allocate structures

% Temporal structures
tld.tmp.conf = zeros(1,tld.nGrid);
tld.tmp.patt = zeros(tld.model.num_trees,tld.nGrid);

overlap     = bb_overlap(tld.bb(:,I),tld.grid); % bottleneck

% Target (display only)
tld.target = img_patch(tld.img{I}.input,tld.bb(:,I));

% Generate Positive Examples
[pX,pEx,bbP] = tldGeneratePositiveData(tld,overlap,tld.img{I},tld.p_par_init);
pY = ones(1,size(pX,2));
% disp(['# P patterns: ' num2str(size(pX,2))]);
% disp(['# P patches : ' num2str(size(pEx,2))]);

% Correct initial bbox
tld.bb(:,I) = bbP(1:4,:);

% Variance threshold
tld.var = var(pEx(:,1))/2;
% disp(['Variance : ' num2str(tld.var)]);

% Generate Negative Examples
[nX,nEx] = tldGenerateNegativeData(tld,overlap,tld.img{I});
% disp(['# N patterns: ' num2str(size(nX,2))]);
% disp(['# N patches : ' num2str(size(nEx,2))]);

% Split Negative Data to Training set and Validation set
[nX1,nX2,nEx1,nEx2] = tldSplitNegativeData(nX,nEx);
nY1  = zeros(1,size(nX1,2));

% Generate Apriori Negative Examples
%[anX,anEx] = tldGenerateAprioriData(tld);
%anY = zeros(1,size(anX,2));
% disp(['# apriori N patterns: ' num2str(size(anX,2))]);
% disp(['# apriori N patches : ' num2str(size(anEx,2))]);

tld.pEx{I}  = pEx; % save positive patches for later
tld.nEx{I}  = nEx; % save negative patches for later
tld.X{I}    = [pX nX1];
tld.Y{I}    = [pY nY1];
idx         = randperm(size(tld.X{I},2));
tld.X{I}    = tld.X{I}(:,idx);
tld.Y{I}    = tld.Y{I}(:,idx);

% Train using training set ------------------------------------------------

% Fern
bootstrap = 2;
fern(2,tld.X{I},tld.Y{I},tld.model.thr_fern,bootstrap);

% Nearest Neightbour 

tld.pex = [];
tld.nex = [];

tld = tldTrainNN(pEx,nEx1,tld);
tld.model.num_init = size(tld.pex,2);

% Estimate thresholds on validation set  ----------------------------------

% Fern
conf_fern = fern(3,nX2);
tld.model.thr_fern = max(max(conf_fern)/tld.model.num_trees,tld.model.thr_fern);

% Nearest neighbor
conf_nn = tldNN(nEx2,tld);
tld.model.thr_nn = max(tld.model.thr_nn,max(conf_nn));
tld.model.thr_nn_valid = max(tld.model.thr_nn_valid,tld.model.thr_nn);