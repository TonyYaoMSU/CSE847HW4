%% Setup parameters
data_file = 'data.txt';
labels_file = 'labels.txt';

data = load(data_file);
data = [data , ones(size(data,1),1)]; % <-- Add bias
labels = load(labels_file);
training_data_sizes = [200, 500, 800, 1000, 1500, 2000]';
accuracy = zeros(size(training_data_sizes));
w = zeros(numel(training_data_sizes), size(data,2));
p = zeros(numel(training_data_sizes), (4601-2000)); % pred

%% Train the logistic regressor on different training data sizes & get accuracies
data_test = data(2001:4601,:);
labels_test = labels(2001:4601);
for i = 1:numel(training_data_sizes)
    % Train the logistic regressor
    n = training_data_sizes(i);
    data_train = data(1:n,:);
    labels_train = labels(1:n);
    weights = logistic_train(data_train, labels_train);
    w(i,:) = weights;
    
    % Compute the predicted values and the testing accuracy
    predictions = 1*(sigmoid(data_test * weights)>0.5);
    p(i,:) = predictions;
    accuracy(i) = sum(predictions == labels_test)/2601;
end

%% Plot the results
figure;
plot(training_data_sizes, accuracy, 'o-');
title('Problem 1: Logistic Regression');
xlabel('n (training data size)');
ylabel('Testing Accuracy');
