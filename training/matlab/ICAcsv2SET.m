clc;clear all;close all;
asrtime = zeros(80, 6);
for i=65:65
    % read data
    filename = [int2str(65), '_ICA_DLtrain.set'];
    EEG = pop_loadset('filename', filename, 'filepath', './setfile');
    eeglab redraw;
    % change data
    %if i==77
    %    i = 71;
    %end
    %decode = ['D:\test\decode\decode_', int2str(i), '_ICA_DLtrain.csv']
    %decode = ['G:\共用雲端硬碟\CNElab_張功逸_Cary_EEGARNet\4.dataset\Real EEG\1. Chronic Migraine (old)\final_result\1DResCNN\',  int2str(i), 'ERP_out.csv']
    EEG.etc.DL = 0;
    output = importdata('./65_output.csv');
    %output = convertStringsToChars(output);        
    EEG.data = output;
    eeglab redraw;
    % channel location
    %EEG.chanlocs = readlocs('C:\Users\User\Downloads\decodecsv\ch19.ced')
    %EEG = pop_eegfiltnew(EEG, 0.5, 50);
    c1 = round(clock);
    %EEG = clean_rawdata(EEG, 5, [0.25 0.75], 0.85, 4, 5, 0.25);
    %EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion','off','ChannelCriterion','off','LineNoiseCriterion','off','Highpass','off','BurstCriterion',20,'WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
    c2 = round(clock);
    asrtime(i,:) = c2 - c1;
    %eeglab redraw;
    % run ICA
    EEG = pop_runica(EEG, 'icatype', 'runica', 'options', {'extended', 1, 'maxsteps', 1024, 'stop', 1e-7});
    diary off;
    eeglab redraw
    % save set file
    EEG = pop_saveset(EEG, 'filename', [int2str(i) '_ICA_unet1.set'], 'filepath', './');
    %filename = ['G:\共用雲端硬碟\CNElab_張功逸_Cary_EEGARNet\4.dataset\P300\ASR\', output{i, 1}]
    %csvwrite(filename, EEG.data);
end

for i=1:76
    filename = [int2str(i), '_ICA_decode.set'];
    EEG = pop_loadset('filename', filename, 'filepath', './decode_set');
    eeglab redraw;
    
    diary off;
    eeglab redraw
    
    clear data icawinv icaact idx_label
    icaact = (EEG.icaweights*EEG.icasphere)*EEG.data;
    icawinv.Original = EEG.icawinv;
    EEG = iclabel(EEG);
    label = strrep((EEG.etc.ic_classification.ICLabel.classes)', ' ', '');
    class = EEG.etc.ic_classification.ICLabel.classifications;
    for k = 1:size(label, 1)
        icawinv.(label{k, 1}) = icawinv.Original;
        
        if strcmp(label{k, 1}, 'Brain')
            idx = (class(:, k) < 0.8)';
            icawinv.(label{k, 1})(:, idx) = 0;
            data.(label{k, 1}) = icawinv.(label{k, 1})*icaact;
            
            
        else
            [idx_label(:,1), idx_label(:,2)] = max(class, [], 2);
            if ~isempty(find(idx_label(:,2)==k, 1))
                icawinv.(label{k, 1})(:,(idx_label(:,2)~=k)') = 0;
                data.(label{k, 1}) = (icawinv.Brain + icawinv.(label{k, 1}))*icaact;
                
            else
                icawinv.(label{k, 1}) = zeros(size(icawinv.Original));
%                 data.(label{k, 1}) = (icawinv.Brain + icawinv.(label{k, 1}))*icaact;
            end
        end           
    end
    EEG.etc.DL.trainingData = data;
    EEG.etc.DL.icawinv = icawinv;
    EEG.etc.DL.icaact = icaact;
   
    
    data_struct = fieldnames(data) 
    for j = 1:size(data_struct)
        if strcmp(string(data_struct(j)), 'Brain')
            outfile = append('./csvdata/Brain/', int2str(i), '.csv')
            csvwrite(outfile, data.Brain)
        elseif strcmp(string(data_struct(j)), 'Muscle')
            outfile = append('./csvdata/Muscle/', int2str(i), '.csv')
            csvwrite(outfile, data.Muscle)
        elseif strcmp(string(data_struct(j)), 'Eye')
            outfile = append('./csvdata/Eye/', int2str(i), '.csv')
            csvwrite(outfile, data.Eye)
        elseif strcmp(string(data_struct(j)), 'Heart')
            outfile = append('./csvdata/Heart/', int2str(i), '.csv')
            csvwrite(outfile, data.Heart)
        elseif strcmp(string(data_struct(j)), 'LineNoise')
            outfile = append('./csvdata/LineNoise/', int2str(i), '.csv')
            csvwrite(outfile, data.LineNoise)
        elseif strcmp(string(data_struct(j)), 'ChannelNoise')
            outfile = append('./csvdata/ChannelNoise/', int2str(i), '.csv')
            csvwrite(outfile, data.ChannelNoise)
        elseif strcmp(string(data_struct(j)), 'Other')
            outfile = append('./csvdata/Other/', int2str(i), '.csv')
            csvwrite(outfile, data.Other)
        end
    end
end