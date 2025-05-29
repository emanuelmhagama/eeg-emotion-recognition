#%% md
# ### Imports
#%%
import scipy.io as io
import numpy as np
import os
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import kurtosis, skew
from datetime import datetime
import traceback
#%% md
# ### Loading Data
#%%
mat_dir = 'data/' 

# List mat files in the mat directory that is not label or readme 
excluded_files = ('label', 'readme', '.')
mat_files = [file for file in os.listdir(mat_dir) if not file.startswith(excluded_files)]

# Helper function to extract and convert the filenames to float, for easy sorting
def extract_numeric_filename(filename):
    return float(filename.replace('_','').replace('.mat',''))  # Convert to an integer

# Sorting the files based on the numeric value
mat_files.sort(key=extract_numeric_filename)

# Taking one experiment for each participant
seen_prefixes = set()
exp_one_files = []
exp_one_data = []

for file in mat_files:
    prefix = file.split('_')[0].strip()
    if prefix not in seen_prefixes:
        seen_prefixes.add(prefix)
        
        # Loading the participant file
        mat_data = io.loadmat(mat_dir+file) 
        
        # Extracting eeg data for 15 trials
        for i in range(1, 16):
            key = f'eeg{i}' # since the mat file keys are suffixed by eeg1 to eeg15
            # Check for keys with the suffix
            for k in mat_data.keys():
                if k.endswith(key):
                    exp_one_data.append(np.array(mat_data[k]))
                    
        # Saving the filenames of the files used
        exp_one_files.append(file)
#%% md
# ### Saving Data to CSV
# For Fast and Easy loading of datasets
#%%
def save_np_to_csv(np_array, name):
    """ Saves numpy array with 3D shape to 2D shape data as csv file, """
    
    reshaped = np_array.reshape(np_array.shape[0] * np_array.shape[1], np_array.shape[2])
    print(f'{datetime.now()} : Saving {name}')
    np.savetxt(f'{name}.csv', reshaped, delimiter=',')
    print(f'{datetime.now()} : Saved {name}')
    
def save_array_to_csv(array, name):
    """ Saves array with 3D shape to 2D shape data as csv file, """
    
    combined_data = np.vstack(array)
    print(f'{datetime.now()} : Saving {name}')
    np.savetxt(f'{name}.csv', combined_data, delimiter=',')
    print(f'{datetime.now()} : Saved {name}')
    
def read_csv_to_np(name):
    """ Reads the csv file and changes it to 3D numpy array """
    
    print(f'{datetime.now()} : Reading {name}')
    np_data = np.loadtxt(f'{name}.csv', delimiter=',')
    reshaped = np_data.reshape(225 ,62, np_data.shape[1])
    print(f'{datetime.now()} : Read {name}')
    return reshaped

def read_csv_to_4d_np(name):
    """ Reads the csv file and changes it to 4D numpy array """

    print('{} : Reading 4D numpy {}'.format(datetime.now(), name))
    reshaped = []
    try:
        np_data = np.loadtxt('{}.csv'.format(name), delimiter=',')
        print('{} : Shape {}'.format(datetime.now(), np_data.shape))
        reshaped = np_data.reshape(np_data.shape[1], 4, 62, np_data.shape[0] // 248)
        print('{} : Read 4D numpy {}'.format(datetime.now(), name))
    except Exception as e:
        print('{} : Error reading 4D numpy: {}'.format(datetime.now(), e))
        traceback.print_exc()
    return reshaped

def save_4d_np_to_csv(np_array, name):
    """ Saves numpy array with 4D shape to 2D shape data as csv file, ()"""
    
    reshaped = np_array.reshape(np_array.shape[1] * np_array.shape[2] * np_array.shape[3], np_array.shape[0])
    print('{} : Saving 4D numpy {}'.format(datetime.now(), name))
    np.savetxt('{}.csv'.format(name), reshaped, delimiter=',')
    print('{} : Saved 4D numpy {}'.format(datetime.now(), name))


#%% md
# ### Plotting the Data
#%% md
# #### Padding the datapoints
# Since the datapoints number varies for different channels, I decided to add paading to the short channels for easy plotting
#%%
max_length = 0
for trial in exp_one_data:
    max_length = max(max_length, trial.shape[1])

exp_one_data_padded = []
for trial in exp_one_data:
    padded_data = np.pad(trial, ((0, 0), (0, max_length - trial.shape[1])), mode='constant', constant_values=0)
    exp_one_data_padded.append(padded_data)

exp_one_data_padded_np =  np.array(exp_one_data_padded)
print(exp_one_data_padded_np.shape)
#save_np_to_csv(exp_one_data_padded_np, 'exp_one_data_padded_np')

# Printing the first and second sample
print(f'Sample 1 shape : {exp_one_data_padded[0].shape}')
print(f'Sample 2 shape : {exp_one_data_padded[1].shape}')
#%% md
# #### Plotting for channel 1
#%%
# Assuming `data` is your EEG data with shape (trials, channels, time_points)
def plot_eeg(data, title):
    plt.figure(figsize=(10, 5))
    plt.plot(data.T)  # Plotting the first channel
    plt.title(title)
    plt.xlabel('Time Points')
    plt.ylabel('Voltage (µV)')
    plt.savefig(f'{title}.png')
    plt.show()

#%%
#exp_one_data_padded_np = read_csv_to_np('exp_one_data_padded_np')
print(f'{datetime.now()} : Plotting  exp_one_data_padded_np')
plot_eeg(exp_one_data_padded_np[:,0,:], 'EEG Data for Channel 1')
print(f'{datetime.now()} : Plotted  exp_one_data_padded_np')
#%% md
# ### Butterworth Band-pass filter
#%%
# Define the sampling frequency (in Hz) and the EEG data
Fs = 200
 
# Define frequency bands
bands = {
    'Alpha': (1, 7),
    'Beta': (8, 13),
    'Theta': (14, 30),
    'Gamma': (30, 45)
}

# Butterworth bandpass filter function
def butter_bandpass(lowcut, highcut, Fs, order=4):
    nyquist = 0.5 * Fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

# Function to apply the filter
def apply_filter(data, lowcut, highcut, Fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, Fs, order)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

# Apply filters to each band, trial, and channel (accounting for variable datapoint lengths)
filtered_data = {band: [] for band in bands}

for band, (lowcut, highcut) in bands.items():
    j = 1
    for trial_data in exp_one_data_padded_np:
        j=j+1
        trial_filtered = []
        i = 1
        for channel_data in trial_data:
            print(f'Processing {band} Band, trial {j} - channel {i} .....')
            i=i+1
            filtered_channel = apply_filter(channel_data, lowcut, highcut, Fs)
            trial_filtered.append(filtered_channel)
        trial_filtered_np = np.array(trial_filtered)
        filtered_data[band].append(trial_filtered_np)  # Store filtered trial
#%%
for band, trials in filtered_data.items():
    save_array_to_csv(trials, f'{band}')
#%% md
# ### Plotting Bands
#%%
alpha_np = read_csv_to_np('Alpha')
plot_eeg(alpha_np[:,0,:], f'EEG Data for Alpha Band Channel 1')
#%%
beta_np = read_csv_to_np('Beta')
plot_eeg(beta_np[:,0,:], f'EEG Data for Beta Band Channel 1')
#%%
theta_np = read_csv_to_np('Theta')
plot_eeg(theta_np[:,0,:], f'EEG Data for Theta Band Channel 1')
#%%
gamma_np = read_csv_to_np('Gamma')
plot_eeg(gamma_np[:,0,:], f'EEG Data for Gamma Band Channel 1')
#%% md
# ### Taking only 80s
# Due to different data lengths of different channels for different samples, we take only the first 80s of the data
# 
# **80s = 200 * 80 = 16,000 data points**
#%%
alpha_np_80s = alpha_np[:,:,:16000]
beta_np_80s = beta_np[:,:,:16000]
theta_np_80s = theta_np[:,:,:16000]
gamma_np_80s = gamma_np[:,:,:16000]

print(f'Alpha {alpha_np_80s.shape}, Beta {beta_np_80s.shape}, Theta {theta_np_80s.shape}, Gamma {gamma_np_80s.shape}')
    
#%% md
# ### Sliding Windows and Feature Matrices
#%%
# Function to create sliding windows
def create_feature_matrix(data, window_size, stride):
    windows = []
    pcc_matrices = []
    mean_features = []
    variance_features = []
    kurtosis_features = []
    skewness_features = []

    # Loop over each trial
    for trial_data in data:
        # Loop through the data with the sliding window
        for start in range(0, trial_data.shape[1] - window_size + 1, stride):
            end = start + window_size
            # Extract the window
            window = trial_data[:, start:end]
            windows.append(window)

            #Extracting PCC features
            pcc_matrices.append(np.corrcoef(window))

            # Extracting statistical features across samples for each channel
            mean_val = np.mean(window, axis=1)
            mean_features.append(mean_val)

            variance_val = np.var(window, axis=1)
            variance_features.append(variance_val)

            kurtosis_val = kurtosis(window, axis=1)
            kurtosis_features.append(kurtosis_val)

            skewness_val = skew(window, axis=1)
            skewness_features.append(skewness_val)

    sc_matrix = np.stack((np.array(mean_features), np.array(variance_features), np.array(kurtosis_features),
                          np.array(skewness_features)), axis=2)
    return np.array(windows), np.array(pcc_matrices), sc_matrix


# Define time window sizes and strides
window_size_8s = 1600  # 8 seconds -> (16000 / 80) * 8 -> 1600 time steps
stride_4s = 800  # 4 seconds -> 800 time steps

window_size_12s = 2400  # 12 seconds -> 2400 time steps
stride_8s = 1600  # 8 seconds -> 1600 time steps

# Create Alpha feature matrices
print('{} : Creating Alpha Feature Matrices'.format(datetime.now()))
alpha_window_matrix_8s, alpha_pcc_matrix_8s, alpha_sc_matrix_8s = create_feature_matrix(
    alpha_np_80s, window_size_8s,
    stride_4s)
alpha_window_matrix_12s, alpha_pcc_matrix_12s, alpha_sc_matrix_12s = create_feature_matrix(
    alpha_np_80s, window_size_12s,
    stride_8s)
print('{} : Created Alpha Feature Matrices'.format(datetime.now()))

# Create Beta feature matrices
print('{} : Creating Beta Feature Matrices'.format(datetime.now()))
beta_window_matrix_8s, beta_pcc_matrix_8s, beta_sc_matrix_8s = create_feature_matrix(beta_np_80s, window_size_8s,
                                                                                     stride_4s)
beta_window_matrix_12s, beta_pcc_matrix_12s, beta_sc_matrix_12s = create_feature_matrix(beta_np_80s, window_size_12s,
                                                                                        stride_8s)
print('{} : Created Beta Feature Matrices'.format(datetime.now()))

# Create Theta feature matrices
print('{} : Creating Theta Feature Matrices'.format(datetime.now()))
theta_window_matrix_8s, theta_pcc_matrix_8s, theta_sc_matrix_8s = create_feature_matrix(
    theta_np_80s, window_size_8s,
    stride_4s)
theta_window_matrix_12s, theta_pcc_matrix_12s, theta_sc_matrix_12s = create_feature_matrix(
    theta_np_80s, window_size_12s,
    stride_8s)
print('{} : Created Theta Feature Matrices'.format(datetime.now()))

# Create Gamma feature matrices
print('{} : Creating Gamma Feature Matrices'.format(datetime.now()))
gamma_window_matrix_8s, gamma_pcc_matrix_8s, gamma_sc_matrix_8s = create_feature_matrix(
    gamma_np_80s, window_size_8s,
    stride_4s)
gamma_window_matrix_12s, gamma_pcc_matrix_12s, gamma_sc_matrix_12s = create_feature_matrix(
    gamma_np_80s, window_size_12s,
    stride_8s)
print('{} : Created Gamma Feature Matrices'.format(datetime.now()))

# Combining the four bands
window_matrix_8s = np.stack((alpha_window_matrix_8s, beta_window_matrix_8s, theta_window_matrix_8s,
                             gamma_window_matrix_8s), axis=1)
window_matrix_12s = np.stack((alpha_window_matrix_12s, beta_window_matrix_12s, theta_window_matrix_12s,
                              gamma_window_matrix_12s), axis=1)
pcc_matrix_8s = np.stack((alpha_pcc_matrix_8s, beta_pcc_matrix_8s, theta_pcc_matrix_8s, gamma_pcc_matrix_8s), axis=1)
pcc_matrix_12s = np.stack((alpha_pcc_matrix_12s, beta_pcc_matrix_12s, theta_pcc_matrix_12s, gamma_pcc_matrix_12s),
                          axis=1)

sc_matrix_8s = np.stack((alpha_sc_matrix_8s, beta_sc_matrix_8s, theta_sc_matrix_8s, gamma_sc_matrix_8s), axis=1)
sc_matrix_12s = np.stack((alpha_sc_matrix_12s, beta_sc_matrix_12s, theta_sc_matrix_12s, gamma_sc_matrix_12s), axis=1)
#%%
#Saving feature matrices
save_4d_np_to_csv(window_matrix_8s, 'windows_matrix_8s')
save_4d_np_to_csv(window_matrix_12s, 'windows_matrix_12s')
save_4d_np_to_csv(pcc_matrix_8s, 'pcc_matrix_8s')
save_4d_np_to_csv(pcc_matrix_12s, 'pcc_matrix_12s')
save_4d_np_to_csv(sc_matrix_8s, 'sc_matrix_8s')
save_4d_np_to_csv(sc_matrix_12s, 'sc_matrix_12s')

# Output feature matrix shapes
print("Window matrix (8s window, 4s stride):", window_matrix_8s.shape)
print("window matrix (12s window, 8s stride):", window_matrix_12s.shape)

print("PCC Feature matrix (8s window, 4s stride):", pcc_matrix_8s.shape)
print("PCC Feature matrix (12s window, 8s stride):", pcc_matrix_12s.shape)

print("SC Feature matrix (8s window, 4s stride):", sc_matrix_8s.shape)
print("SC Feature matrix (12s window, 8s stride):", sc_matrix_12s.shape)
#%% md
# ### PCA
#%%
window_matrix_12s = read_csv_to_4d_np('windows_matrix_12s')
window_matrix_8s = read_csv_to_4d_np('windows_matrix_8s')

def get_pca_features(data):
    print('{} : Creating PCA Feature Matrices'.format(datetime.now()))
    # We need to reshape data so that time points are processed correctly.
    # Let's combine windows, bands, and channels, leaving the temporal dimension intact.

    # Flatten segments, bands, and channels together: (segments * bands * channels, datapoints)
    reshaped_data = data.reshape(-1, data.shape[3])

    # Apply PCA on temporal dimension (2400 datapoints)
    n_components_time = 62  # Desired number of components
    pca = PCA(n_components=n_components_time)

    # Apply PCA
    print('{} : Applying PCA ...'.format(datetime.now()))
    pca_result = pca.fit_transform(reshaped_data)

    # Reshape back to original dimensionality without temporal reduction: (segments, 4, 62, n_components_time)
    pca_matrix = pca_result.reshape(data.shape[0], 4, 62, n_components_time)

    print('{} : Created PCA Feature Matrices, Shape {}'.format(datetime.now(), pca_matrix.shape))
    return pca_matrix
pca_matrix_8s = get_pca_features(window_matrix_8s)
pca_matrix_12s = get_pca_features(window_matrix_12s)

print("PCA Feature matrix (8s window, 4s stride):", pca_matrix_8s.shape)
print("PCA Feature matrix (12s window, 8s stride):", pca_matrix_12s.shape)

save_4d_np_to_csv(pca_matrix_8s, 'pca_matrix_8s')
save_4d_np_to_csv(pca_matrix_12s, 'pca_matrix_12s')

print("********************** {} : FINISHED *********************".format(datetime.now()))