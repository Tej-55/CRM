clc
clear

%load train file from CRM structure data
simple_feats_tr = load('train.csv');
simple_feats_te = load('test.csv');

%load crm prediction ensemble
crm_ens_preds_tr = load('feats_tr.csv');
crm_ens_preds_te = load('feats.csv');

%number of input features (simple features in CRM str)
num_simple_feats = 371;

%number of ensembles
num_ens = size(crm_ens_preds_tr,2)-1;

%create train and test dataset files
train_data = zeros(size(simple_feats_tr,1), num_simple_feats+num_ens+1);
test_data = zeros(size(simple_feats_te,1), num_simple_feats+num_ens+1);

train_data(:,1:num_simple_feats) = simple_feats_tr(:,1:num_simple_feats);
test_data(:,1:num_simple_feats) = simple_feats_te(:,1:num_simple_feats);

train_data(:,num_simple_feats+1:num_simple_feats+num_ens) = crm_ens_preds_tr(:,2:num_ens+1);
test_data(:,num_simple_feats+1:num_simple_feats+num_ens) = crm_ens_preds_te(:,2:num_ens+1);

for i = 1:size(train_data,1)
    if mode(crm_ens_preds_tr(i,2:num_ens+1)) == crm_ens_preds_tr(i,1)
        train_data(i,end) = 1;
    else
        train_data(i,end) = 0;
    end
end

for i = 1:size(test_data,1)
    if mode(crm_ens_preds_te(i,2:num_ens+1)) == crm_ens_preds_te(i,1)
        test_data(i,end) = 1;
    else
        test_data(i,end) = 0;
    end
end

%save the train and test files for MLP
csvwrite('mlp_train_data.csv', train_data);
csvwrite('mlp_test_data.csv', test_data);