%%
figure(1)
dense_FE = load("loss\DenseNet_feature_extraction.mat");
%dense_FT = load("loss\DenseNet_fine_tuning.mat");
plot(dense_FE.train_loss);
hold on
plot(dense_FE.val_loss);
hold on
title('DenseNet');
legend('Training Loss', 'Validation Loss', 'Location', 'Best');
xlabel('Epoch');
ylabel('Loss');
print('plots\densenet_loss.eps', '-depsc', '-r300');

%%
figure(2)
google_FE = load("loss\GoogLeNet_feature_extraction.mat");
plot(dense_FE.train_loss);
hold on
plot(dense_FE.val_loss);
hold on
title('GoogLeNet');
legend('Training Loss', 'Validation Loss', 'Location', 'Best');
xlabel('Epoch');
ylabel('Loss');
print('plots\googlenet_loss.eps', '-depsc', '-r300');