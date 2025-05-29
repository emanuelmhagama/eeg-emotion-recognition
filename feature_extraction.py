#%%
# -*- coding: utf-8 -*-
#%% md
# ### Imports
#%%
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from datetime import datetime
from sklearn.decomposition import PCA
import traceback

#%% md
# ### Loading Data
#%%
print("************************* {} : STARTING FEATURE EXTRACTION **********************".format(datetime.now()))

#%%
def save_np_to_csv(np_array, name):
    """ Saves numpy array with 3D shape to 2D shape data as csv file, """

    reshaped = np_array.reshape(np_array.shape[0] * np_array.shape[1], np_array.shape[2])
    print('{} : Saving {}'.format(datetime.now(), name))
    np.savetxt('{}.csv'.format(name), reshaped, delimiter=',')
    print('{} : Saved {}'.format(datetime.now(), name))


def save_4d_np_to_csv(np_array, name):
    """ Saves numpy array with 4D shape to 2D shape data as csv file, ()"""
    
    reshaped = np_array.reshape(np_array.shape[1] * np_array.shape[2] * np_array.shape[3], np_array.shape[0])
    print('{} : Saving 4D numpy {}'.format(datetime.now(), name))
    np.savetxt('{}.csv'.format(name), reshaped, delimiter=',')
    print('{} : Saved 4D numpy {}'.format(datetime.now(), name))


def read_csv_to_np(name):
    """ Reads the csv file and changes it to 3D numpy array """

    print('{} : Reading {}'.format(datetime.now(), name))
    np_data = np.loadtxt('{}.csv'.format(name), delimiter=',')
    reshaped = np_data.reshape(225, 62, np_data.shape[1])
    print('{} : Read {}'.format(datetime.now(), name))
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


#%% md
# # ### Plotting the Data
# 
# # Assuming `data` is your EEG data with shape (trials, channels, time_points)
# def plot_eeg(data, title):
#     plt.figure(figsize=(10, 5))
#     plt.plot(data.T)  # Plotting the first channel
#     plt.title(title)
#     plt.xlabel('Time Points')
#     plt.ylabel('Voltage (µV)')
#     plt.savefig('{}.png'.format(title))
#     plt.show()
# 
# 
#%% md
# # ### Plotting Bands
#%%
alpha_np = read_csv_to_np('Alpha')
#plot_eeg(alpha_np[:, 0, :], 'EEG Data for Alpha Band Channel 1')
#%%
beta_np = read_csv_to_np('Beta')
#plot_eeg(beta_np[:, 0, :], 'EEG Data for Beta Band Channel 1')
#%%
theta_np = read_csv_to_np('Theta')
#plot_eeg(theta_np[:, 0, :], 'EEG Data for Theta Band Channel 1')
#%%
gamma_np = read_csv_to_np('Gamma')
#plot_eeg(gamma_np[:, 0, :], 'EEG Data for Gamma Band Channel 1')
#%% md
# # ### Taking only 80s
# # Due to different data lengths of different channels for different samples, we take only the first 80s of the data
# # 
# # **80s = 200 * 80 = 16,000 data points**
#%%
alpha_np_80s = alpha_np[:, :, :16000]
beta_np_80s = beta_np[:, :, :16000]
theta_np_80s = theta_np[:, :, :16000]
gamma_np_80s = gamma_np[:, :, :16000]

print('Alpha {}, Beta {}, Theta {}, Gamma {}'.format(alpha_np_80s.shape, beta_np_80s.shape, theta_np_80s.shape,
                                                     gamma_np_80s.shape))


#%% md
# # ### Sliding Windows and Feature Matrices
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

save_4d_np_to_csv(pca_matrix_8s,'pca_matrix_8s')
save_4d_np_to_csv(pca_matrix_12s, 'pca_matrix_12s')
#%%
print("********************** {} : FINISHED *********************".format(datetime.now()))