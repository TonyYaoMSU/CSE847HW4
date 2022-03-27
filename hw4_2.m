
addpath(fullfile('SLEP-master','SLEP','functions','L1','L1R'));
addpath(fullfile('SLEP-master','SLEP','functions'));
addpath(fullfile('SLEP-master','SLEP','opts'));
addpath(fullfile('SLEP-master','SLEP','CFiles'));
addpath(fullfile('SLEP-master','SLEP'));
addpath(fullfile('SLEP-master'));
data_file = 'ad_data.mat';
data = load(data_file);
data_train = data.X_train;
data_test = data.X_test;
label_train = data.y_train;
label_test = data.y_test;
par  = [0.00001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
w = zeros(numel(par),size(data_train,2)); % all weights
b = zeros(numel(par),1); % all biases
p = zeros(numel(par), size(data_test,1)); % all predictions
aucs = zeros(size(par));
num_feats = zeros(size(par));
for i = 1:numel(par)
    
    % Train the logistic regressor
    [weights, bias] = logistic_l1_train(data_train, label_train, par(i));
    w(i,:) = weights;
    b(i) = bias;
    
    % Count the number of nonzero features
    num_feats(i) = sum(weights ~= 0);
    
    % Compute the predicted values
    predictions = 1*(sigmoid(data_train * weights + bias )> 0.5);

    [~,~,~,auc] = perfcurve(label_train, predictions,1);
    aucs(i) = auc;
end

figure;
plot(par, aucs, 'o-');
title('Problem 2: Sparse Logistic Regression');
xlabel('l_1 regularization parameter');
ylabel('Area Under Curve (AUC)');

figure;
plot(par, num_feats, 'o-');
title('Problem 2: Sparse Logistic Regression');
xlabel('l_1 regularization parameter');
ylabel('Number of Features');