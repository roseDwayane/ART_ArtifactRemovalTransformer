clc;clear all;close all;
MSEtabel = zeros(5, 30);
for i=65:65
    model_file = {'_ICA_DLtrain.set', '_ICA_unetpp3.set', '_ICA_unetpp4.set', '_ICA_unetpp5.set', '_ICA_decode.set'};
    model_fold = {'./setfile', './unetpp3setfile', './unetpp4setfile', './unetpp5setfile','D:\decode_set'};
    
    for j=1:5
        filename = [int2str(i), model_file{1,j}];
        EEG = pop_loadset('filename', filename, 'filepath', model_fold{1,j});
        EEG.data = normalize(double(EEG.data), 'range', [-1 1]);
        EEG = pop_epoch(EEG, {'251','252'}, [-0.3, 2]);
        EEG = pop_rmbase(EEG, [-300, 0]);
        idx = (EEG.times <= 0);
        eeglab redraw;

        data_lenth = size(EEG.data(1,:,1),2);
        data_double = size(EEG.data(1,:,:),3);

        temp = zeros(30, data_lenth);
        for k=1:data_double
            for m=1:30
                temp(m,:) = temp(m,:) + EEG.data(m,:,k);
            end
        end
        groundtruth = temp/data_lenth;
        
        y1 = zeros(30,data_lenth);
        for k=1:30
            y1_m = mean(groundtruth(k,idx));
            y1_s = std(groundtruth(k,idx));
            y1(k,:) = (groundtruth(k,:) - y1_m) / y1_s;
        end

        y2 = zeros(30,data_lenth);
        for k=1:30
            k=15
            temp = zeros(1,data_double);
            for m=1:data_double
                y2_m = mean(EEG.data(k,idx,m));
                y2_s = std(EEG.data(k,idx,m));
                y2(k,:) = (EEG.data(k,:,m) - y2_m) / y2_s;
                temp(m) = immse(y1(k,:), double(y2(k,:)));
            end
            B = sort(temp(1,:));
            for m=1:data_double
                C = find(temp==B(1,m));
                y2_m = mean(EEG.data(k,idx,C));
                y2_s = std(EEG.data(k,idx,C));
                y2(k,:) = (EEG.data(k,:,C) - y2_m) / y2_s;
                clf;
                X = linspace(-0.3, 2, data_lenth);
                plot(X, y1(k,:)); hold on;
                plot(X, y2(k,:));
                axis([-0.3 0.5 -6 6]);
                savefig_name = ['./ERP_snr/', model_file{1,j}, '/ch', int2str(k), 'n', int2str(m), '.png'];
                saveas(gcf, savefig_name)
            end
            MSEtabel(j,k) = mean(temp);
        end
        savetable_name = ['./ERP_snr/', 'total.csv'];
        csvwrite(savetable_name, reshape(MSEtabel,[5,30]));
    end
end
