for i = 65:65
     filename = [int2str(i), '_ICA_DLtrain.set'];
     EEG = pop_loadset('filename', filename, 'filepath', './setfile');
     eeglab redraw;
     
     filename2 = ['./rawcsv', int2str(i), '_ICA_DLtrain.csv'];
     csvwrite(filename2, EEG.data);
end