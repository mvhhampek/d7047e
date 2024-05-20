%% 
res_FE = load("loss\ResNet_feature_extraction.mat");
google_FE = load("loss\GoogLeNet_feature_extraction.mat");
dense_FE = load("loss\DenseNet_feature_extraction.mat");
res_FT = load("loss\ResNet_fine_tuning.mat");
google_FT = load("loss\GoogLeNet_fine_tuning.mat");
dense_FT = load("loss\DenseNet_fine_tuning.mat");

%% FEATURE EXTRACTION TRAINING LOSS
figure(1)
plot(res_FE.train_loss);
hold on
plot(google_FE.train_loss);
hold on
plot(dense_FE.train_loss);
hold off

legend('ResNet', ...
   'GoogLeNet', ...
   'DenseNet', ...
   'Location', 'Best', ...
   'FontSize', 12);

upper_y = max(res_FE.train_loss(1), google_FE.train_loss(1));
upper_y = max(upper_y, dense_FE.train_loss(1))*1.05;
lower_y = min(res_FE.val_loss)*0.95;
ylim([lower_y, upper_y]);

xlim([1 28]);

xlabel('Epoch',  'FontSize', 12);
ylabel('Loss',  'FontSize', 12);
print('plots\training_loss_FE.eps', '-depsc', '-r300');


%% FEATURE EXTRACTION VALIDATION LOSS
figure(2)
plot(res_FE.val_loss);
hold on
plot(google_FE.val_loss);
hold on
plot(dense_FE.val_loss);
hold off

legend('ResNet ', ...
   'GoogLeNet ', ...
   'DenseNet ', ...
   'Location', 'Best', ...
   'FontSize', 12);

upper_y = max(res_FE.val_loss(1), google_FE.val_loss(1));
upper_y = max(upper_y, dense_FE.val_loss(1))*1.05;
lower_y = min(res_FE.val_loss)*0.95;
ylim([lower_y, upper_y]);

xlim([1 28]);

xlabel('Epoch',  'FontSize', 12);
ylabel('Loss',  'FontSize', 12);
print('plots\validation_loss_FE.eps', '-depsc', '-r300');

%% FINE TUNING TRAINING LOSS
figure(1)
plot(res_FT.train_loss);
hold on
plot(google_FT.train_loss);
hold on
plot(dense_FT.train_loss);
hold off

legend('ResNet', ...
   'GoogLeNet', ...
   'DenseNet', ...
   'Location', 'Best', ...
   'FontSize', 12);

upper_y = max(res_FT.train_loss(1), google_FT.train_loss(1));
upper_y = max(upper_y, dense_FT.train_loss(1))*1.05;
lower_y = min(google_FT.val_loss)*0.01;
ylim([lower_y, upper_y]);

xlim([1 15]);

xlabel('Epoch',  'FontSize', 12);
ylabel('Loss',  'FontSize', 12);
print('plots\training_loss_FT.eps', '-depsc', '-r300');


%% FINE TUNING VALIDATION LOSS
figure(1)
plot(res_FT.val_loss);
hold on
plot(google_FT.val_loss);
hold on
plot(dense_FT.val_loss);
hold off

legend('ResNet ', ...
   'GoogLeNet ', ...
   'DenseNet ', ...
   'Location', 'Best', ...
   'FontSize', 12);

upper_y = max(res_FT.val_loss(1), google_FT.val_loss(1));
upper_y = max(upper_y, dense_FT.val_loss(1))*1.05;

ylim([0.04, upper_y]);

xlim([1 15]);

xlabel('Epoch',  'FontSize', 12);
ylabel('Loss',  'FontSize', 12);
print('plots\validation_loss_FT.eps', '-depsc', '-r300');
