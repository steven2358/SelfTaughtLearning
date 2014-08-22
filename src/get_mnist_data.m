function [images,labels] = get_mnist_data(dataset)
% Returns images and labels for training or testing. Checks if data is in
% '/data' folder and downloads from remote repository if necessary.
%
% Original URLS:
% - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
% - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
% - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
% - http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

if nargin<1,
    dataset = 'train';
end

switch dataset
    case 'train'
        fname_data = 'train-images-idx3-ubyte';
        fname_labels = 'train-labels-idx1-ubyte';
    case 'test'
        fname_data = 't10k-images-idx3-ubyte';
        fname_labels = 't10k-labels-idx1-ubyte';
end
base_dir = 'http://yann.lecun.com/exdb/mnist/';
url_data = [base_dir fname_data '.gz'];
url_labels = [base_dir fname_labels '.gz'];

data_dir = '../data/';

%% load data

if (exist([data_dir fname_data],'file') ~= 2)
    t1 = tic;
    fprintf('Downloading %s images... ',dataset);
    gunzip(url_data,data_dir);
    fprintf('%.2fs.\n',toc(t1));
end

if (exist([data_dir fname_labels],'file') ~= 2)
    t2 = tic;
    fprintf('Downloading %s labels... ',dataset);
    gunzip(url_labels,data_dir)
    fprintf('%.2fs.\n',toc(t2));
end

t3 = tic;
fprintf('Loading %s images... ',dataset);
images = loadMNISTImages([data_dir fname_data]);
fprintf('%.2fs.\n',toc(t3));

t4 = tic;
fprintf('Loading %s labels... ',dataset);
labels = loadMNISTLabels([data_dir fname_labels]);
fprintf('%.2fs.\n',toc(t4));
