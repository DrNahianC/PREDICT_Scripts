%%This is the script for the main data analysis pipeline. Data was loaded
%%at the point where bad epochs were rejected (see sensor_PAF_pipeline) and then 
%ICA was run to manually select the sensorimotor component 


clear all
close all
wpms.DATAOUT    = 'X:\Schabrun group data\PREDICT\Analysis\EEG\ICA_test\Processed\Nahian\'; % Change to wherever data is stored
subjlist = dir([wpms.DATAOUT 'sub-*']);
exp.sessions  = {'ses-00','ses-02','ses-05'};

%Add functions
addpath('X:\Schabrun group data\Chowdhury Nahian\EEGLAB_FieldTrip\eeglab_current\eeglab2019_1')
addpath(genpath('X:\Schabrun group data\Chowdhury Nahian\EEGLAB_FieldTrip\eeglab_current\eeglab2019_1\functions'));
addpath('X:\Schabrun group data\Chowdhury Nahian\EEGLAB_FieldTrip\eeglab_current\eeglab2019_1\plugins\xdfimport1.16');
addpath('X:\Schabrun group data\Chowdhury Nahian\EEGLAB_FieldTrip\eeglab_current\eeglab2019_1\plugins\bva-io1.5.13');
addpath(genpath('X:\Schabrun group data\Chowdhury Nahian\EEGLAB_FieldTrip\eeglab_current\eeglab2019_1\plugins\PrepPipeline0.55.3'));
addpath(genpath('X:\Schabrun group data\Chowdhury Nahian\EEGLAB_FieldTrip\eeglab_current\eeglab2019_1\plugins\firfilt'));
addpath('X:\Schabrun group data\Chowdhury Nahian\EEGLAB_FieldTrip\fieldtrip-20200215');
addpath('X:\Schabrun group data\Chowdhury Nahian\EEGLAB_FieldTrip\fieldtrip-20200215\external\eeglab'); 

% Load Channel locations
load('X:\Schabrun group data\Chowdhury Nahian\chanlocs.mat') % change to wherever channel locations are stored
final_data = "";



%% FIELDTRIP
clearvars all_eeg_files EEG n_triggers prompt rejected triggers px day
for px = 1:length(subjlist);
    clearvars all_eeg_files cfg data data_rejected data_freq data_pruned EEG freq paf power promt n_rejected rejected temp temp2 X y z
    %clearvars datasets
    fprintf(['\n Analysing participant: ' subjlist(px).name '\n\n']);
    this_subject = subjlist(px).name;
    all_eeg_files  = dir([wpms.DATAOUT subjlist(px).name filesep '*_processed_eeglab_fieldtrip.mat']);
  
    for day = 1:length(all_eeg_files);
       % if exist([wpms.DATAOUT subjlist(px).name filesep subjlist(px).name '_' exp.sessions{day} '_paf_componentspace.mat'],'file');
       %     fprintf('\n Have already analyzed this participant \n\n');
       %  else

            clearvars cfg data data_freq data_comp EEG freq paf power promt rejected temp temp2 X y z n_rejected;
            load([all_eeg_files(day).folder filesep all_eeg_files(day).name], 'data_rejected', 'EEG');
            
            %% ICA and Plot Components

            %Run ICA
            cfg          = [];
            cfg.method = 'runica';
            data_comp = ft_componentanalysis(cfg, data_rejected);

            %Create plots of the components
            cfg           = [];
            cfg.component = [1:20];       % specify the component(s) that should be plotted
            cfg.layout    = 'EEG1010.lay'; % specify the layout file that should be used for plotting
            cfg.viewmode = 'component';
            cfg.comment   = 'no';
            f = figure;
            movegui(f,[0 300]);
            ft_topoplotIC(cfg, data_comp);
            cfg.position = [700 300 500 500];
            ft_databrowser(cfg, data_comp);

            % Spectral Plots of Components

            cfg              = [];
            cfg.output       = 'pow';
            cfg.channel      = 'all';%compute the power spectrum in all ICs
            cfg.method       = 'mtmfft';
            cfg.taper        = 'hanning';
            cfg.foi          = [2:.20:50];
            data_freq = ft_freqanalysis(cfg, data_comp);

            nsubplots = 25;
            nbyn = sqrt(nsubplots);% sqrt(nsubplots) should not contain decimals,
            type doc subplot;
            Nfigs = ceil(size(data_comp.topo,1)/nsubplots);
            tot = Nfigs*nsubplots;
            rptvect = 1:size(data_comp.topo,1);
            rptvect = padarray(rptvect, [0 tot-size(data_comp.topo,1)], 0,'post');
            rptvect = reshape(rptvect,nsubplots,Nfigs)';
            for r=1:1;
                f = figure
                %movegui(f,[1400 300]);
                k=0;
                for j=1:20;
                    if~(rptvect(r,j)==0);
                        k=k+1;
                        cfg=[];
                        cfg.channel = rptvect(r,j);
                        subplot(nbyn,nbyn,k);ft_singleplotER(cfg,data_freq);
                         windows = {[9:.2:11],[8:.2:12]};
                         window_indicies = {[36:46],[31:51]};
                         freq_window = 2;
                         temp = transpose(data_freq.powspctrm(j,:));
                         temp2 = zscore(temp);
                         spectral_data(:,j) = temp;
                         paf_this_component(:,j) = sum(windows{freq_window}'.*(((temp(window_indicies{freq_window})))))/sum(((((temp(window_indicies{freq_window}))))));
                         
                    end
                end
            end

            %% Choose Sensorimotor Component
            fprintf(['\n Waiting for component selection for: ' subjlist(px).name '\n\n']);
            prompt = 'Please identify the sensorimotor component';
            chosen_component = input(prompt);
            close all;
            %% Calculate PAF for chosen component
            chosen_component_paf = paf_this_component(chosen_component);
            chosen_component_data = data_comp.topo(:,chosen_component);
            chosen_spectral_data = spectral_data(:,chosen_component);
           
            %save data
            save(['X:\Schabrun group data\Chowdhury Nahian\PREDICT - Projects\Main Outcomes Paper\Component_Data\' subjlist(px).name  '_component_data.mat'],'data_comp','data_freq','chosen_component_data','chosen_component_paf', 'chosen_spectral_data','spectral_data');% Nahian - Adjust based on what you th
            %save topoplots of chosen component
            map = topoplot(chosen_component_data,EEG.chanlocs, 'headrad', 0.66, 'plotrad', 0.72);
            title(string(chosen_component_paf))
            saveas(map,['X:\Schabrun group data\Chowdhury Nahian\PREDICT - Projects\Main Outcomes Paper\Component_Data\' subjlist(px).name '_topoplot'], 'png');% Nahian - Adjust based on what you think data output should be called
            %save spectral plot of chosen component 
            spectral = plot([2:.20:50],spectral_data(:,chosen_component));
            title(string(chosen_component_paf))
            saveas(spectral,['X:\Schabrun group data\Chowdhury Nahian\PREDICT - Projects\Main Outcomes Paper\Component_Data\' subjlist(px).name '_spectralplot'], 'png');% Nahian - Adjust based on what you think data output should be called

            %store chosen component data
            final_data(px,1) = subjlist(px).name;
            final_data(px,2) = string(chosen_component_paf);
            final_data(px,3) = string(chosen_component);

            %%his code will create matrices containing component level data for
            %%each channels while  accounting for missing channels, so that they can be
            %%collated across participants 
            chosen_component_data(:,2) = chosen_component_data(:,1);
            for i = 1:63;
                channels(i,:) = string(chanlocs(i).labels);
            end
            channels_after_exclusion = string(data_2_badtrials.label');

            missing_channels = channels(~all(ismember(channels,channels_after_exclusion),2),:);
            for j = 1:length(missing_channels);
                index_missing_channels(j) = find(channels==missing_channels(j));
            end

            for j = 1:length(missing_channels)
                chosen_component_data = [chosen_component_data(1:(index_missing_channels(j)-1), :); nan(1,size(chosen_component_data,2)); chosen_component_data(index_missing_channels(j):end, :)];
            end

            chosen_component_data = string(chosen_component_data);
            chosen_component_data(:,1) = channels;
            save(['X:\Schabrun group data\Chowdhury Nahian\PREDICT - Projects\Main Outcomes Paper\Component_Data\' subjlist(px).name  '_components_all_channels.mat'],'chosen_component_data')


            close all;
       % end
    end
    fprintf(['\n Finished analysing participant: ' subjlist(px).name '\n\n']);
end







