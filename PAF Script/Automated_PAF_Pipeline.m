%% Purpose:
%%This is the script for the automated component selection analysis pipeline. Data was loaded
%%at the point where bad epochs were rejected (see sensor_PAF_pipeline),
%%and then ICA was run followed by an automated component selection to
%%reduce subjectivity associated with the manual component selection 

%This script uses Ali Mazehari's alpha IC isolating toolbox function to determine an
%ICA component which displays an alpha peak and most closely follows a
%topography indicative of a dipole in the sensorimotor cortex, according to
%Ali's template.

%It applies a 0.2 to 30 hz bandpass filter to the data (see lines 45 and
%56) And excludes noisy channels + ECG.
%This version constrains the ICA in a PCA first.

%Sarah_template_2 is a template generated from data in
%/data/Andrew/ali_eeg/AliSpatialICA. It is very similar to the sensorimotor
%alpha component published in Furman et al, Neuroimage. 2018 (fig2)

clear all
close all

%% Set-up Workspace %%
cwd = [pwd filesep]; % Store current working directory
wpms = []; % pre-allocate a workspace variables in struct

wpms.DATAIN    = [cwd 'Output' filesep];
wpms.DATAOUT    = [cwd 'Output_AutomatedICA' filesep];
wpms.FUNCTIONS    = [cwd 'eeglab_fieldtrip' filesep];
wpms.TEMPLATE = [cwd 'channel_info' filesep]

subjlist = dir([wpms.DATAIN 'sub-*']);
exp.sessions  = {'ses-00','ses-02','ses-05'};


%Sort these functions out
template_home = wpms.TEMPLATE


addpath([wpms.FUNCTIONS '\fieldtrip-20200215'])
addpath([wpms.FUNCTIONS '\fieldtrip-20200215\external\eeglab'])
addpath([wpms.FUNCTIONS '\fieldtrip-20200215\utilities'])


for px = 1:length(subjlist);
    clearvars -except subjlist wpms exp template_home b f px num_its
    fprintf(['\n Analysing participant: ' subjlist(px).name '\n\n']);
    this_subject = subjlist(px).name;
    %load sensor level data 
    all_eeg_files  = dir([wpms.DATAIN subjlist(px).name filesep '*_processed_eeglab_fieldtrip.mat']);

    for f = 1:1
        num_its = [1 2 3 4 5 6 7 8 9 10]
        for b = 1:length(num_its)

            clearvars -except subjlist all_eeg_files wpms exp template_home b f px num_its

            iter = num_its(b);
            iter = num2str(iter)

            run = all_eeg_files(f).name;
            
            %load fieldtrip data from the bad trials rejected stage
            load([all_eeg_files(f).folder filesep all_eeg_files(f).name], 'data_rejected');
            data = data_rejected;
            %%%%
            
            %load channels and exclude AFz as this was ground 
            cfg = [];
            cfg.channel = {'all' '-AFz'};
            data = ft_selectdata(cfg, data);

            %Run ICA 
            cfg = [];
            cfg.method ='fastica';
            cfg.runica.pca = 15
            [IC] = ft_componentanalysis(cfg, data);


            %% this is where my very simple function comes in .  It picks a component based on topography
            cd(template_home)
            cfg.layout='EEG1010.lay'; % the layout for the data
            cfg.foi=[8 12] %%% what frequency to chose
            cfg.template='SarahTemplate_2'; %%%  This is the topography you want it to match to the componenent.
            %%%It uses this as a template for testing components that have alpha peak

            %% Function is this bit:
            cfg2       = [];
            cfg2.method = 'mtmfft';
            cfg2.output = 'pow';
            cfg2.pad    = 'maxperlen';
            cfg2.foilim = [1 30];
            cfg2.taper  = 'hanning';
            cfg2.keeptrials='no'
            alpha_IC   = ft_freqanalysis(cfg2, IC);

            %% get bandwith
            bandwidth=find(alpha_IC.freq<=cfg.foi(2)&alpha_IC.freq>=cfg.foi(1));

            f_spectra=(alpha_IC.powspctrm);

            %% use the matlab find peaks to find local maxima (i.e a peak in a bandwidth
            for k=1:size(f_spectra,1),
                [peak{k},locs{k}]=findpeaks( f_spectra(k,:) ) ;
                [value ploc(k)]=max(peak{k});
                where_peak_happen(k)=locs{k}(ploc(k));
            end

            %% cget compoment
            alpha_components=find(ismember(where_peak_happen,bandwidth));

            %%%%%%%%%%%%%load the template file
            load(cfg.template);
            %template=cfg.template;
            %%%% correlate

            %% match electrodes;
            elec=template_electrodes{:};  % basically load electrode positions, compare with template, match order.. blah blah

            [common_electrodes index]=ismember(IC.topolabel,elec);

            size(template);
            template_electrodes=elec(index);

 
            template=template(index);

            %%% correlate the electrodes in the data with the electrodes in the
            %%% template
            for k=1:length(alpha_components),

                [r(k) p(k)]  =corr(template, IC.topo(:,alpha_components(k)));

            end

            [max_r maximum_component_index]=(max(abs(r)));

            selected_component=alpha_components(maximum_component_index);
            cfg.zlim='absmax';
            cfg.component=selected_component;
            ft_topoplotIC(cfg,IC); title(strcat('correlation prob between IC and template= ',num2str(p(maximum_component_index))));;
            colormap('jet');
            savefig([wpms.DATAOUT subjlist(px).name '_iter_' char(string(b)) '_topoplot.fig'])

            dummy=IC;
            dummy.dimord='chan_time'
            dummy.time=1;
            dummy.temp=template;
            cfg.parameter='temp';
            cfg.zlim='absmax';
            cfg.comment            = 'no';
            figure
            ft_topoplotER(cfg,dummy);title('template');colormap('jet');

            %% end function

            %  cd(file_home)
            %%%Sarah's stuff added here:
            %extract the correct component - go back to original IC and select data
            %from just the winner.
            cfg3       = [];
            cfg3.method = 'mtmfft';
            cfg3.output = 'pow';
            cfg3.pad    = 'maxperlen';
            cfg3.foilim = [1 30];
            cfg3.taper  = 'hanning';
            cfg3.keeptrials='yes' %% this allows us to get a value for each trial (epoch), IC, and frequency
            % cfg3.pad = 4;
            alpha_IC_trials   = ft_freqanalysis(cfg3, IC);


            %now in alpha_IC_trials.powspctrm have trial*comp*frequency- we only care
            %about 1 comp.
            allfreq_IC_timeseries = alpha_IC_trials.powspctrm(:,selected_component, :);
            %this is now a 2D mat with a power value per trial/frquency bin.

            %lifted from existing preprocessing script
            PAF = [];
            PAF_Timeseries = [];
            Alpha_Power = [];
            Power_Timeseries=[];
            freq = 8:.2:12;

            temp = [];
            temp = squeeze(allfreq_IC_timeseries(:,1,:));; %generates matrix of frequency bin power values over time
            %temp = zscore(temp);
            for i = 1:size(allfreq_IC_timeseries,1); %for each epoch

                %29:45 corresponds to 8-12 Hz
                temp_paf(i) = sum(freq.*temp(i,36:56))/sum(temp(i,36:56)); %Cog method
                temp_power(i) = sum(temp(i,36:56));  %corresponds to 8-12

            end

            PAF = mean(temp_paf);
            PAF_Timeseries= temp_paf; %generates a time-series of 4 sec epochs
            Alpha_Power = mean(temp_power); %avg over the entire scan
            Power_Timeseries = temp_power; %generates a time-series of 4 sec epochs


            corr_to_template = (p(maximum_component_index))

            selected_component=alpha_components(maximum_component_index);

            %cd(file_sub)

            %cfg.zlim='absmax';
            %cfg.component=selected_component;
            %ft_topoplotIC(cfg,IC)
            %title(strcat('correlation prob between IC and template= ',num2str(p(maximum_component_index))));;
            %colormap('jet');

            save([wpms.DATAOUT subjlist(px).name '_iter_' char(string(b)) '_sarah_ICA.mat'],'IC', 'corr_to_template', 'alpha_components', 'selected_component','alpha_IC','PAF','PAF_Timeseries', 'p', 'Alpha_Power', 'cfg', 'cfg2', 'cfg3', 'Power_Timeseries','alpha_IC_trials', 'maximum_component_index', 'max_r')

            %this variable ^^ "shows my work" update to the name/location of your script or delete it.
            close all
        end
    end

end


%% DECIDE WHICH ITERATION GENERATES THE BEST CORRELATION

subjlist = dir([wpms.DATAOUT '*_sarah_ICA.mat']);

file_count = 1;
for sub = 1:length(subjlist)/10;
    clearvars m index
    for this_file = 1:10;
        clearvars max_r PAF alpha_components alpha_IC_trials IC selected_component
        load([subjlist(file_count).folder filesep subjlist(file_count).name], 'IC', 'alpha_components', 'alpha_IC_trials', 'selected_component');
        fprintf(['\n Analysing participant: ' subjlist(file_count).name '\n\n']);
        cd(template_home)
        cfg.template='SarahTemplate_2';

        load(cfg.template);
        %template=cfg.template;
        %%%% correlate

        %% match electrodes;
        elec=template_electrodes{:};  % basically load electrode positions, compare with template, match order.. blah blah

        [common_electrodes index]=ismember(IC.topolabel,elec);

        size(template);
        template_electrodes=elec(index);

        %excluded_channels = find(index == 0);
        %IC.topolabel(:,excluded_channels) = []
        %IC.topo(:,excluded_channels) = []

        %for q =1:length(index)
        %    template{q} = elec{index};
        %end
        template=template(index);


        %%% correlate the electrodes in the data with the electrodes in the
        %%% template
        for k=1:length(alpha_components),

            [r(k) p(k)]  =corr(template, IC.topo(:,alpha_components(k)));

        end

        [max_r maximum_component_index]=(max(abs(r)));

        allfreq_IC_timeseries = alpha_IC_trials.powspctrm(:,selected_component, :);

        %lifted from existing preprocessing script
        PAF = [];
        PAF_Timeseries = [];
        Alpha_Power = [];
        Power_Timeseries=[];
        freq = 8:.2:12;

        temp = [];
        temp = squeeze(allfreq_IC_timeseries(:,1,:)); %generates matrix of frequency bin power values over time
        %temp2 = zscore(temp);

        for i = 1:size(allfreq_IC_timeseries,1); %for each epoch
            %29:45 corresponds to 8-12 Hz
            temp_paf(i) = sum(freq.*temp(i,36:56))/sum(temp(i,36:56)); %Cog method
            % temp_paf_z(i) = sum(freq.*temp2(i,36:56))/sum(temp2(i,36:56)); %Cog method
        end

        PAF = mean(temp_paf);
        %PAF_z = mean(temp_paf_z);

        fprintf(['\n PAF is: ' char(string(PAF)) '\n\n']);
        fprintf(['\n Corr is: ' char(string(max_r)) '\n\n']);

        %fprintf(['\n PAF_z is: ' char(string(PAF_z)) '\n\n']);

        PAFs_this_participant(this_file,:) = PAF;
        % PAFs_z_this_participant(this_file, :) = PAF_z;
        correlations_this_participant(this_file,:) = max_r;
        file_count = file_count + 1;
    end
    [m,index] = max(correlations_this_participant);
    PAF_best_match(sub,:) = PAFs_this_participant(index)
    %PAF_z_best_match(sub,:) = PAFs_this_participant(index)
    Best_correlations(sub,:) = m;
end
PAF_best_match(PAF_best_match == 0) = NaN
save([wpms.DATAOUT 'winners.mat'],'PAF_best_match','Best_correlations')

