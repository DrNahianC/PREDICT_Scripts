clear all;
%% Set-up Workspace %%
cwd = [pwd filesep]; % Store current working directory
wpms = []; % pre-allocate a workspace variables in struct

% Locate folders of interest - Change if necessary
wpms.DATAIN     = fullfile(cwd, ['..'  filesep 'PREDICT_Data'], filesep); % /../ skips back a folder
wpms.DATAOUT    = [cwd 'Output' filesep];
wpms.GRIDLOC    = [cwd 'Gridloc' filesep];

% Structure and store subject codes
subjlist = dir([wpms.DATAIN 'sub-*']);

% Prepare OUTPUT folders - !!!! Do this only once to set-up. !!!! Comment
% out afterwards
mkdir(wpms.DATAOUT); % Create Output directory
for subout = 1:length(subjlist)
    mkdir([wpms.DATAOUT subjlist(subout).name]) % Create folder for each px
end

% Experiment variables of interest - structure and store
exp           = [];
exp.sessions  = {'ses-00','ses-02','ses-05'};
exp.tasks     = {'task-dem','task-MVC','task-Hot','task-Thr','task-Map','task-Con'};
exp.filetypes = {'S2R','s2rx','smr','csv','mat'};

%% Run script %%

% Run script looping through each subject code. 'px' is used becuase it
% stands out more within the loop amoung the variable names

for px = 1:length(subjlist) %Change the 1 here if you want start at a specific participant e.g. change 1 to 16
    
    % print out subject codes to review
    fprintf(['\n Analysing participant: ' subjlist(px).name '\n\n']);
    
    % Look for subject data files
    spikefiles  = dir([wpms.DATAIN subjlist(px).name filesep 'tms' filesep '*-Map*.mat']); %the spike files for the mapping procedure
    coordinates = dir([wpms.DATAIN subjlist(px).name filesep 'tms' filesep '*_locations.csv']); % files containing the locations corresponding to each tms pulse 
    
    % Sanity check - should be equal numbers of files
    if length(spikefiles) ~= length(coordinates)
        fprintf('\n Error: The number of spike files does not match the number of coordinates. Please check and correct \n\n')
    end
    
    % Sanity check - should not be greater than three files
    if length(spikefiles) > length(exp.sessions) || length(coordinates) > length(exp.sessions)
        fprintf('\n Error: The number of spike files is greater than the number of sessions. Please check and correct \n\n')
    end
    
    % Loop through sessions, making sure the sessions match and there is
    % not more than one file for each session. 'day' is used becuase it
    % stands out more within the loop among variable names
    
    for day = 1:length(spikefiles) %Change the 1 here if you want start at a specific session e.g. change 1 to 3 for the 3rd session
        
        % print out session codes to review
        fprintf(['\n Analysing session: ' exp.sessions{day} '\n\n']);
        
        % Check this data has not already been analysed. If so skip this
        % participant
        % if exist([wpms.DATAOUT subjlist(px).name filesep subjlist(px).name '_' exp.sessions{day} '_TrialData.mat']) % TODO: Change to what the final data will be called.
        %    fprintf(['\n This participants ' exp.sessions{day} ' data has already been processed. Moving on... \n\n']);
            
        %else
            % Print out exact files to be loaded for QC
            fprintf(['Working on the following spike file: ' spikefiles(day).name '\n'])
            fprintf(['Working on the following coordinate file: ' coordinates(day).name '\n'])

            fprintf('Loading and organising data... \n')
            
            load([wpms.GRIDLOC 'Grid.mat']); %load the grid used for all participants 
            spike_data = load([spikefiles(day).folder filesep spikefiles(day).name]); %load spike data
            loaded_coordinates = readtable([coordinates(day).folder filesep coordinates(day).name],'ReadVariableNames',0); %load locations

     
            %% Organize the Data
            spike_data   = struct2cell(spike_data); %convert spike data to structure 
            n_trials     = length(spike_data{2}.times); %load number of tms pulses
            stimuluated_locations    = table2array(loaded_coordinates(1:n_trials,2)); %locations in a table 
            tms_times    = round(spike_data{2}.times / 0.0005 ) * 0.0005; %round tms times 
            
            %store EMG Data
            emg_data         = [];
            emg_data(:,1)    = spike_data{1}.values; 
            emg_data(:,2)    = spike_data{1}.times;
            
     
            %% Calculate rms of background_emg
            fprintf('Starting rms calculations... \n')
            
            rms_background_emg = [];
            for this_trial = 1:n_trials
                background_emg_window = emg_data((emg_data(:,2)>(tms_times(this_trial)-0.055)) & (emg_data(:,2)<(tms_times(this_trial)-0.005)),:); % isolate background emg between 55 and 5ms before tms pulse
                background_emg_window = abs( background_emg_window); %rectify signal
                rms_background_emg(this_trial,1) = sqrt(mean((background_emg_window(:,1)).^2)); %calculate rms
            end
            
            
            %% Calculate RMS of each MEP window and subtract from background activity
            rms_mep = [];
            for this_mep = 1:n_trials
                trace_this_trial = emg_data((emg_data(:,2)>(tms_times(this_mep)-0.05)) & (emg_data(:,2)<(tms_times(this_mep)+0.05)),:); %obtain a trace relative to tms pulse
                trace_this_trial(:,2) = trace_this_trial(:,2)-tms_times(this_mep); %calibrate time relative to tms pulse
                trace_this_trial(:,1) = abs(trace_this_trial(:,1)); %rectify the signal
                selected_trace_window = trace_this_trial((trace_this_trial(:,2)>=0.0073) & (trace_this_trial(:,2)<=0.0153),:); %masseter mep window is between 7.3 and 15.3ms after pulse
                selected_trace_window_emg_column = selected_trace_window(:,1); %isolate emg column
                rms_mep(this_mep,1) = sqrt(mean((selected_trace_window_emg_column).^2)); %calculate rms
            end
            
            rms = rms_mep - rms_background_emg; %subtract rms of mep window from rms of background emg
            rms(rms<0) = 0; %code all negative values as zero
            
            %% Store all data
            stimuluated_locations  = string(stimuluated_locations);
            final_data = [];
            final_data = [stimuluated_locations,rms(:,1), rms_mep(:,1), rms_background_emg(:,1)];
            %% Remove Repeated or Empty TMS_data
            fprintf('remove repeated or false trials... \n')
            comments =  table2array(loaded_coordinates(1:(n_trials),3));
            repeated_trials = find(((strcmp(comments,'repeat'))));
            false_trials = find(((strcmp(comments,'empty'))));
            nocontract_trials = find(((strcmp(comments,'nocontract'))));
            Unknown_trials = find(((strcmp(comments,'unknown'))));
            all_deleted_trials = transpose(cat(1, repeated_trials, false_trials, nocontract_trials, Unknown_trials));
            if length(all_deleted_trials) > 0
                final_data(all_deleted_trials,:)= [];
            end
        
           
            %% Put mean of each location on a grid
            
            fprintf('Creating Map... \n')
            new_grid = Grid;
            grid_dims = size(new_grid);
            for i = 1:grid_dims(1)
                for j = 1:grid_dims(2)
                    location = Grid(i, j);
                    meps_of_this_location = final_data((final_data(:,1)==location),:);
                    mean_of_this_location = mean(str2double((meps_of_this_location(:,2))));
                    new_grid(i, j) = mean_of_this_location;
                end
            end
            
            converted_Grid = str2double(new_grid);
            converted_Grid(isnan(converted_Grid))=0;
            %If each site is normalized relative to the max MEP
            %Converted_Grid = (Converted_Grid/(max(Converted_Grid,[],'all')))*100;
            
            stored_map_data = array2table(converted_Grid, 'VariableNames', {'-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3','4','5'}, 'RowNames', {'12','11','10','9','8','7','6','5','4','3','2','1','0'});
            
            %% Map Parameters
            
            map_volume = sum(converted_Grid(converted_Grid>max(converted_Grid, [],'all')/10)); %sum 
            active_sites = converted_Grid > max(converted_Grid,[],'all')/10;
            map_area(px, day) = sum(active_sites(:) == 1);
            %of all active sites defined as a site greater than >10% of the
            %maximum MEP amplitude 
            
            participant_volumes(px, day) = map_volume
    
    %end use this for the if else loop for existing folders  
    end % Session number for loop
    
end % Particpant for loop


