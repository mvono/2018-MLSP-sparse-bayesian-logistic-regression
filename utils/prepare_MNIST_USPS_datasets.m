%-------------------------------------------------------------------------%
%                        PREPARE MNIST & USPS DATASETS                    %
%                    FOR THE BINARY CLASSIFICATION PROBLEM                %
%-------------------------------------------------------------------------%

clearvars;

%% MNIST dataset
features1 = loadMNISTImages('dataset/MNIST/train_images')';
labels1 = loadMNISTLabels('dataset/MNIST/train_labels'); 
lab1 = [1 4 3];
lab2 = [7 6 5];
for i = 1:length(lab1)
    idx = find(labels1 == lab1(i) | labels1 == lab2(i));
    features = features1(idx,:);
    labels = labels1(idx);
    save(['dataset/MNIST/ini_' num2str(lab1(i)) num2str(lab2(i)) '.mat'], ...
         'features','labels');
end

% Check that the file is saved
fprintf('MNIST initial parameters saved! \n');
clearvars;

%% USPS dataset
load ./dataset/USPS/USPS
labels1 = gnd;
features1 = fea;
lab1 = [1 4 3];
lab2 = [7 6 5];
for i = 1:length(lab1)
    idx = find(labels1 == lab1(i) | labels1 == lab2(i));
    features = features1(idx,:);
    labels = labels1(idx);
    save(['dataset/USPS/ini_' num2str(lab1(i)) num2str(lab2(i)) '.mat'], ...
         'features','labels');
end

% Check that the file is saved
fprintf('USPS initial parameters saved! \n');
clearvars;