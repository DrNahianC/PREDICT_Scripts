%% This is the full pipeline for extracting sensor-level PAF data. Note that this is also
%the backbone for the Automated and Manual component selection pipleines,
%as the code for the data loading, bad channel selection and bad epoch selection 
%is shown here. The Automated and manual component selection scripts start
%at the point after bad epochs are removed in fieldtrip. 


%% Set-up Workspace %%
cwd = [pwd filesep]; % Store current working directory
wpms = []; % pre-allocate a workspace variables in struct

% Locate folders of interest - Change if necessary
wpms.DATAIN     = [cwd 'PREDICT_EEG_Raw_Data' filesep]; 
wpms.DATAOUT    = [cwd 'Output' filesep];
wpms.FUNCTIONS    = [cwd 'eeglab_fieldtrip' filesep];
wpms.CHANNELS = [cwd 'channel_info' filesep]

%Add functions
addpath([wpms.FUNCTIONS '\eeglab2019_1'])
addpath(genpath([wpms.FUNCTIONS '\eeglab2019_1\functions']));
addpath([wpms.FUNCTIONS '\eeglab2019_1\plugins\xdfimport1.16'])
addpath([wpms.FUNCTIONS '\eeglab2019_1\plugins\bva-io1.5.13'])
addpath(genpath([wpms.FUNCTIONS '\eeglab2019_1\plugins\PrepPipeline0.55.3']));
addpath(genpath([wpms.FUNCTIONS '\eeglab2019_1\plugins\firfilt']));
addpath([wpms.FUNCTIONS '\fieldtrip-20200215'])
addpath([wpms.FUNCTIONS '\fieldtrip-20200215\external\eeglab'])
load([wpms.CHANNELS '\chanlocs.mat'])
load([wpms.CHANNELS '\neighbour_template.mat'])



% Structure and store subject codes
subjlist = dir([wpms.DATAIN 'sub-*']);
exp.sessions  = {'ses-00','ses-02','ses-05'};

mkdir(wpms.DATAIN); % Create Output directory
for subout = 1:length(subjlist)
    mkdir([wpms.DATAOUT subjlist(subout).name]) % Create folder for each px
end

%% PIPELINE

for px = 1:length(subjlist);
    clearvars all_eeg_files EEG end_of_window end_time n_triggers start_of_window triggers 
    
    fprintf(['\n Analysing participant: ' subjlist(px).name '\n\n']);
    %load eeg files for participant
    all_eeg_files  = dir([wpms.DATAIN subjlist(px).name filesep 'eeg' filesep '*eeg*.vhdr']);   
    for day = 1:length(all_eeg_files);
        clearvars EEG end_of_window end_time n_triggers start_of_window triggers 
        fprintf(['\n Analysing session: ' exp.sessions{day} '\n\n']);
        [EEG, ~] = pop_loadbv(all_eeg_files(day).folder, [all_eeg_files(day).name]); %load session data
        EEG = pop_resample(EEG, 500); %downsample 500Hz
        %lookup channels
        EEG = pop_chanedit(EEG, 'lookup', [wpms.FUNCTIONS 'eeglab2019_1\plugins\dipfit\standard_BESA\standard-10-5-cap385.elp']);
        % remove auxilliary channels
        EEG = pop_select(EEG, 'nochannel', {'GSR','HR','RESP'});
        EEG = pop_reref(EEG, []); %average reference data
        EEG = pop_eegfiltnew(EEG, 2, 100, 826, 0, [], 1); %filter the data 
        close all;
        
        %This code will go through  and find the number of triggers 
        % we pressed to indicate the start and end on the recording. There should be
        %two.. If there are two, we will use the 5 minutes window 3 seconds
        %after we pressed start. If we did not indicate the start and end
        %triggers, we manually idenfiied the end time for the recording,
        %and selected the time window 5 minutes prior to the end point. 
        for i = 1:length(EEG.event);
            triggers{i} = EEG.event(1, i).type;
              
        end
        triggers = string(triggers);
        n_triggers =  length(triggers(triggers == 'Start/End'));

        if n_triggers == 2;
            EEG.event(1, find( triggers == "Start/End", 1)).type = 'Start';
            EEG = pop_rmdat(EEG, {'Start'},[3 303], 0);
        else
            EEG.event(1, (length(EEG.event)+1)).type = 'End';
            EEG.event(1, end).latency = length(EEG.times);
            pop_eegplot(EEG, 1, 1, 1);
            prompt = 'Please identify the end time';
            end_time = input(prompt);
            end_of_window = end_time - length(EEG.times)/500;
            start_of_window = end_of_window - 300;
            EEG = pop_rmdat(EEG, {'End'},[start_of_window end_of_window], 0);
        end
        
        close all 

        % Here we plotted the channels in time and frequency domains to
        % identify any bad channels
        figure; pop_spectopo(EEG, 1, [0 15000], 'EEG', 'freqrange', [2 100], 'electrodes','off');
        pop_eegplot(EEG, 1, 1, 1);
        prompt = 'Please identify sensor # to remove:';
        rejected = input(prompt);       
        rejected_channels{px, day} = rejected;
        
        %Re-reference the data if any channels were rejected 
        if size(rejected,1)~=0;
            EEG = pop_select(EEG, 'nochannel',rejected);  
            EEG = pop_reref(EEG, []);
        end
        figure; pop_spectopo(EEG, 1, [0 15000], 'EEG', 'freqrange', [2 100], 'electrodes','off');
        
        %save the eeglab processed data
        save([wpms.DATAOUT subjlist(px).name filesep subjlist(px).name '_' exp.sessions{day} '_processed_eeglab.mat'], 'EEG','rejected_channels');
        fprintf(['\n Finished analyzing session: ' exp.sessions{day} '\n\n']);
    end
    fprintf(['\n Finished analysing participant: ' subjlist(px).name '\n\n']);
end

%FIELDTRIP 
clearvars all_eeg_files EEG n_triggers prompt rejected subjlist triggers px day 
subjlist = dir([wpms.DATAOUT 'sub-*']);
for px = 1:length(subjlist)
    clearvars all_eeg_files cfg data data_freq data_pruned EEG freq paf power promt rejected temp temp2 X y z
    all_eeg_files  = dir([wpms.DATAOUT subjlist(px).name filesep '*_processed_eeglab.mat']);
    for day = 1:length(all_eeg_files);
        clearvars cfg data data_freq data_pruned EEG freq paf power promt rejected temp temp2 X y z
        %load eeglab data
        load([all_eeg_files(day).folder filesep all_eeg_files(day).name], 'EEG');
        %convert EEGlab to fieldtrip
        data = eeglab2fieldtrip(EEG, 'preprocessing');
        data.label = {EEG.chanlocs.labels};
        
        %cut 300 seconds into pieces of 5 seconds
        cfg = [];
        cfg.length = 5;
        cfg.overlap = 0;
        data=ft_redefinetrial(cfg,data);
        
        %reject bad epochs
        cfg          = [];
        cfg.method   = 'trial';
        cfg.alim = 100;
        data_rejected = ft_rejectvisual(cfg,data);
        n_rejected = length(data.trial) - length(data_rejected.trial);
        
        %Run ICA with 15 components to observe
        cfg          = [];
        cfg.method = 'runica';
        cfg.runica.pca = 15;
        X = ft_componentanalysis(cfg,data_rejected);
        cfg.component = [1:15];
        
        %Create plots of the components
        cfg.layout = 'EEG1010.lay';
        ft_topoplotIC(cfg,X);
        ft_databrowser(cfg,X);
        
        %Reject Visual Components
        prompt = 'Please Identify Component ro reject in matrix form';
        rejected = input(prompt);
        cfg.component = rejected;
        data_pruned = ft_rejectcomponent(cfg,X);
        n_rejected_components= length(rejected)

        %interpolate missing channels
        cfg= [];
        cfg.neighbours = neighboursCopy;
        cfg.method         = 'nearest'
        cfg.layout = 'EEG1010.lay';
        cfg.missingchannel = {neighboursCopy((~ismember({neighboursCopy(:).label}, data_pruned.label))).label};
        data_repaired  = ft_channelrepair(cfg, data_pruned)
        
        %Perform frequency decomposition 
        cfg = [];
        cfg.method = 'mtmfft';
        cfg.taper = 'hanning';
        cfg.foi = [2:.20:50];
        cfg.keeptrials = 'yes';
        data_freq = ft_freqanalysis(cfg, data);
        close all
        
        %Calculate PAF using a 9-11Hz window, with bins of .2 Hz
        paf = [];
        power = [];
        freq = 9:.2:11;
        for z = 1:size(data_freq.powspctrm,1);
            for y = 1:size(data_freq.powspctrm,2);
                temp = squeeze(data_freq.powspctrm(z,y,:));
                temp2 = zscore(temp);
                paf(y,z) = sum(freq'.*(((temp(36:46)))))/sum(((((temp(36:46))))));
                power(y,z) = sum(temp(31:51));
            end
        end
        save([wpms.DATAOUT subjlist(px).name filesep subjlist(px).name '_' exp.sessions{day} '_processed_eeglab_fieldtrip.mat'],'paf','power','data','data_freq','data_repaired','data_pruned','data_rejected');
    end
end



