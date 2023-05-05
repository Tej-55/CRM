clc;
clear;

disp('***Baseline Model for Synthetic Dataset***')

problem = 'chess';
NumTrials = 1000;

%% Parameters for different problems
if strcmp(problem,'trains')
    %CRM graph (node connection list)
    conn_list = dlmread('./trains_crm_graph/rand_crm_1np.pl');
    
    depth3_range = [232, 977];
    depth2_range = [23, 231];
    depth1_range = [1, 22];
    target_theory = [327];

    D_tr = load('./../Winnow2/processed/trains/train.csv'); 
    D_te = load('./../Winnow2/processed/trains/test.csv');
else %chess
    conn_list = dlmread('./chess_crm_graph/rand_crm_d3v6r1.pl');
    
    depth3_range = [162, 412];
    depth2_range = [25, 161];
    depth1_range = [1, 24];
    target_theory = [1,4,48,70];

    D_tr = load('./../Winnow2/processed/chess/train.csv'); 
    D_te = load('./../Winnow2/processed/chess/test.csv');
end

%% Construct CRM graph 
adj = [];
j = 0;
for i = 1:size(conn_list,1)
    if conn_list(i,2) ~= 0
        j = j+1;
        adj(j,:) = [conn_list(i,1),conn_list(i,2)];
    end

    if conn_list(i,3) ~= 0
        j = j+1;
        adj(j,:) = [conn_list(i,1),conn_list(i,3)];
    end
end
G = digraph(adj(:,1), adj(:,2));

%% majority class predictor
[~,c] = size(D_tr);

num_class1_inst = length(find(D_tr(:,c) == 1));
num_class0_inst = length(find(D_tr(:,c) == 0));

if num_class0_inst >= num_class1_inst
    maj_class = 0;
else
    maj_class = 1;
end

fprintf('Majority class: %d\n',maj_class);

[r,c] = size(D_te);

%% Accuracy calculation
%some counters
tp_count = 0;
fn_count = 0;
tn_count = 0;
fp_count = 0;
for i = 1:r
    if D_te(i,c) == 1
        if maj_class == 1
            tp_count = tp_count + 1;
        else
            fn_count = fn_count + 1;
        end
    else
        if maj_class == 0
            tn_count = tn_count + 1;
        else
            fp_count = fp_count + 1;
        end
    end
end
%calculate accuracy score
acc = (tp_count + tn_count)/(tp_count + fp_count + tn_count + fn_count);
fprintf('Accuracy: %f\n',acc);


%% Fidelity calculation
Fidelity = zeros(1,NumTrials);

rng(0);

%How many random trials
for trial = 1:NumTrials
    %some counters
    cep_count = 0;  %correctly explained positives
    cen_count = 0;  %correctly explained negatives
    iep_count = 0;  %incorrectly explained positives
    ien_count = 0;  %incorrectly explained negatives
    
    for i = 1:r
        %which neurons have f=1 at the output layer
        O_f1 = find(D_te(1,depth3_range(1):depth3_range(2)) == 1) + depth3_range(1) - 1;
        
        %draw a neuron from the output layer
        drawn = randi([O_f1(1) O_f1(end)]);
    
        %if prediction is 1
        if D_te(i,c) == 1
            if ~isempty(intersect(bfsearch(G,drawn),target_theory))
                cep_count = cep_count + 1;
            else
                iep_count = iep_count + 1;
            end
        else
            if ~isempty(intersect(bfsearch(G,drawn),target_theory))
                ien_count = ien_count + 1;
            else
                cen_count = cen_count + 1;
            end
        end
    end
    %calculate fidelity and store
    Fidelity(trial) = (cep_count + cen_count) / (cep_count + cen_count + iep_count + ien_count);
end
fprintf('Avg. Fidelity: %f\n',mean(Fidelity));
