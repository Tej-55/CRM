clc
clear

%load train file from CRM structure data
simple_feats_tr = load('train.csv');
simple_feats_te = load('test.csv');

%load crm prediction ensemble
crm_ens_preds_tr = load('feats_tr.csv');
crm_ens_preds_te = load('feats.csv');

%number of input features (simple features in CRM str)
num_simple_feats = 1373;

%number of ensembles
num_ens = size(crm_ens_preds_tr,2)-1;

%create train and test dataset files
train_data = zeros(size(simple_feats_tr,1), num_simple_feats+2*num_ens);
test_data = zeros(size(simple_feats_te,1), num_simple_feats+2*num_ens);

train_data(:,1:num_simple_feats) = simple_feats_tr(:,1:num_simple_feats);
test_data(:,1:num_simple_feats) = simple_feats_te(:,1:num_simple_feats);

train_data(:,num_simple_feats+1:num_simple_feats+num_ens) = crm_ens_preds_tr(:,2:num_ens+1);
test_data(:,num_simple_feats+1:num_simple_feats+num_ens) = crm_ens_preds_te(:,2:num_ens+1);

for i = 1:size(train_data,1)
    for j = 1:num_ens
        if crm_ens_preds_tr(i,j+1) == crm_ens_preds_tr(i,1)
            train_data(i,num_simple_feats+num_ens+j) = 1; %CRM was correct
        else
            train_data(i,num_simple_feats+num_ens+j) = 0; %CRM was incorrect
        end
    end
end

for i = 1:size(test_data,1)
    for j = 1:num_ens
        if crm_ens_preds_te(i,j+1) == crm_ens_preds_te(i,1)
            test_data(i,num_simple_feats+num_ens+j) = 1; %CRM was correct
        else
            test_data(i,num_simple_feats+num_ens+j) = 0; %CRM was incorrect
        end
    end
end

%save the train and test files for MLP
csvwrite('mlp_multilabel_train_data.csv', train_data);
csvwrite('mlp_multilabel_test_data.csv', test_data);
