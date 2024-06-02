import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#------------------------------------------------------------------------------#
def sae_filter(df, filter_params=None):
    # BUTTERWORTH 4-POLE PHASELESS DIGITAL FILTER
    # According to SAE J211-1
    # Written by
    # Daniel Lanner
    # 8 march 2006
    if filter_params is None:
        filter_params = {}
    ytot = df.copy()
    for col in df.columns:
        if col in filter_params:
            CFC, T = filter_params[col]
        else:
            CFC, T = 180, 0.001
        wd = 2 * np.pi * CFC * 2.0775
        wa = np.sin(wd * T / 2) / np.cos(wd * T / 2)
        a0 = wa ** 2 / (1 + np.sqrt(2) * wa + wa ** 2)
        a1 = 2 * a0
        a2 = a0
        b1 = -2 * (wa ** 2 - 1) / (1 + np.sqrt(2) * wa + wa ** 2)
        b2 = (-1 + np.sqrt(2) * wa - wa ** 2) / (1 + np.sqrt(2) * wa + wa ** 2)
        x = df[col].values
        y1 = np.array([x[0], x[1]])
        for i in range(2, len(x)):
            y1 = np.append(y1, a0 * x[i] + a1 * x[i - 1] + a2 * x[i - 2] + b1 * y1[i - 1] + b2 * y1[i - 2])
        y1 = np.flip(y1)
        y = np.array([y1[0], y1[1]])
        for j in range(2, len(y1)):
            y = np.append(y, a0 * y1[j] + a1 * y1[j - 1] + a2 * y1[j - 2] + b1 * y[j - 1] + b2 * y[j - 2])
        y = np.flip(y)
        ytot[col] = y
    return ytot

#------------------------------------------------------------------------------#
def differentiate_rotational_velocity(df, dt, window_size):
    # Apply a 5-point moving average filter to the rotational velocity components
    df_smooth = df[['rv_x', 'rv_y', 'rv_z']].rolling(window_size, center=True).mean()

    # Calculate the resultant rotational velocity for each time step
    df_smooth['rv_r'] = np.sqrt(df_smooth['rv_x']**2 + df_smooth['rv_y']**2 + df_smooth['rv_z']**2)

    # Differentiate the smoothed rotational velocities using NumPy's gradient function
    velocities = df_smooth[['rv_x', 'rv_y', 'rv_z']].values  # convert velocity columns to NumPy array
    accelerations = np.gradient(velocities, axis=0) / dt  # axis=0 specifies that we're differentiating along the rows

    # Add the acceleration components to the dataframe
    df['ra_x'] = accelerations[:, 0]
    df['ra_y'] = accelerations[:, 1]
    df['ra_z'] = accelerations[:, 2]
    df['ra_r'] = np.sqrt(df['ra_x']**2 + df['ra_y']**2 + df['ra_z']**2)

    return df

#------------------------------------------------------------------------------#
# Reading in data file
base_path = 'C:/Users/ameli/OneDrive - Imperial College London/Documents/Schoolwork/Imperial/Masters/Data/Impacts - Copy/ImpactResults/'
# Reading in list from 'ClenchedID.txt'
input_data = pd.read_csv(base_path+'ImpactID.txt')

# Getting a list of all helmet names
test_names = input_data['ImpactID'].tolist()

# Getting the folders associated with the helmet list
folders = []
for j in np.arange(0, len(test_names), 1):
    folders.append(test_names[j][11:-14])
    test_names[j] = test_names[j]+'.csv'

print('Analysing ', len(input_data), 'tests...')

# set all filter settings here:
dt = 1/20000
CFC1000 = (1650, dt)
CFC600 = (1000, dt)
CFC180 = (300, dt)
lin_acc_filter = CFC600
lin_acc_filter_str = 'CFC600'
rot_vel_filter = CFC180
rot_vel_filter_str = 'CFC180'
n_moving_ave = 1
diff_str = '_1pDiff'
rot_processing_str = str(n_moving_ave)+'_pDiff'

# Looping through ALL
for fileIDidx in np.arange(0, len(input_data), 1):
    path = base_path + folders[fileIDidx] + '/'
    i = test_names[fileIDidx]
    print(i)

    # reading in data and processing it...
    newHeader = ['Time (s)',  'rv_z', 'rv_y', 'rv_x','la_z', 'la_y', 'la_x'] #new headform
    #newHeader = ['Time (s)', 'la_x', 'la_y', 'la_z' ,'rv_x', 'rv_y', 'rv_z']
    data = pd.read_csv(path+i, header=None, skiprows=23, usecols=range(0, 7), names=newHeader)
    # Translating units... (degrees to radians)
    data['rv_x'] = np.radians(data['rv_x']) #invert x component for new headform
    data['rv_y'] = np.radians(data['rv_y']) #invert y component for new headform
    data['rv_z'] = -np.radians(data['rv_z']) #invert z component for new headform

    # calculating dt
    dt = data['Time (s)'][1]-data['Time (s)'][0]

    # Taking the region of interest
    data = data.loc[((data['Time (s)'] > -0.04) & (data['Time (s)'] < 0.1))]

    # filter
    filter_params = {'la_x': lin_acc_filter, 'la_y': lin_acc_filter, 'la_z': lin_acc_filter, 'rv_x': rot_vel_filter, 'rv_y': rot_vel_filter, 'rv_z': rot_vel_filter}
    filtered_data = sae_filter(data, filter_params=filter_params)
    
    # determining resultant components
    filtered_data['la_r'] = np.sqrt(filtered_data['la_x']**2 + filtered_data['la_y']**2 + filtered_data['la_z']**2)
    filtered_data['rv_r'] = np.sqrt(filtered_data['rv_x']**2 + filtered_data['rv_y']**2 + filtered_data['rv_z']**2)

    # Save the filtered data to an Excel file
    filtered_data.to_excel('filtered_data.xlsx', index=False)


    # differentiate
    diff_filtered_data = differentiate_rotational_velocity(filtered_data, dt, window_size=n_moving_ave)
    diff_filtered_data['ID'] = i.replace('_UNFILTERED.csv','')
    diff_filtered_data['TestID'] = i[-21:-18]

    # # Changing into krad/s2 
    # diff_filtered_data['ra_x'] = diff_filtered_data['ra_x']/1000
    # diff_filtered_data['ra_y'] = diff_filtered_data['ra_y']/1000
    # diff_filtered_data['ra_z'] = diff_filtered_data['ra_z']/1000
    # diff_filtered_data['ra_r'] = diff_filtered_data['ra_r']/1000

    # Getting the average offset and accounting for it
    for z in ['la_x', 'la_y', 'la_z', 'la_r', 'rv_x', 'rv_y', 'rv_z', 'ra_r', 'ra_x', 'ra_y', 'ra_z', 'ra_r']:
        offset = diff_filtered_data.loc[((diff_filtered_data['Time (s)'] > -0.008) & (diff_filtered_data['Time (s)'] < -0.003))][z].mean()
        diff_filtered_data[z] = diff_filtered_data[z] - offset
        print(f"Offset for {z}: {offset}")
        print(diff_filtered_data)

    # Saving out filtered data
    suffix = 'PROCESSED_'+lin_acc_filter_str+diff_str
    rot_suffix = 'PROCESSED_'+rot_vel_filter_str+diff_str
    file_name = i.replace('UNFILTERED',suffix)
#        print(path+file_name)

    diff_filtered_data.to_csv(path+file_name, index=False)

    # Adding all data into one array
    if i == test_names[0]:
        all_data = diff_filtered_data
    else:
        all_data = pd.concat([all_data,diff_filtered_data])


all_peak_values = pd.DataFrame(None)

# Making numeric columns numeric
all_data.apply(pd.to_numeric, errors='coerce').fillna(all_data)

# Getting absolute values of components
all_abs_data = all_data
#    peak_values = all_abs_data.loc[all_data['ID'] == all_data['ID'].drop_duplicates().tolist()[0]].max()
##    print((peak_values))
#    exit()
#    all_abs_data[x] = all_abs_data[x].abs()
#    Retrieve the values with the corresponding signs
#    result = df.lookup(max_magnitude, max_magnitude.index)

# Extracting peak values from the all_data DataFrame for each value of ID
peak_values_df = pd.DataFrame(columns=['ID', 'la_x', 'la_y', 'la_z', 'la_r', 'rv_x', 'rv_y', 'rv_z', 'rv_r', 'ra_x', 'ra_y', 'ra_z', 'ra_r'])


#    all_data=all_data.head(15)
# print(all_data)


for ival in all_data['ID'].drop_duplicates().tolist():
    row = {'ID': ival, 'TestID': ival[-6:-3]}
    for x in ['la_x', 'la_y', 'la_z', 'la_r', 'rv_x', 'rv_y', 'rv_z', 'rv_r', 'ra_x', 'ra_y', 'ra_z', 'ra_r']:
        # Find the index associated with the max magnitude for each 'ID' and 'x'
        max_magnitude_idx = all_data.loc[all_data['ID'] == ival, x].abs().idxmax()
        # Get the corresponding peak value
        if len(all_data['ID'].drop_duplicates().tolist()) > 1:
            peak_value = all_data.loc[max_magnitude_idx, x].iloc[0]
        else:
            peak_value = all_data.loc[max_magnitude_idx, x]
        # Check if the row exists in peak_values_df, if not add a new row
        if ival not in peak_values_df['ID'].values:
            peak_values_df = peak_values_df.append(row, ignore_index=True)
        # Assign the peak value to the corresponding cell in the peak_values_df DataFrame
        peak_values_df.at[peak_values_df[peak_values_df['ID'] == ival].index[0], x] = peak_value



#    print(peak_values)
    # Peak values is a series...
#        all_peak_values[i] = peak_values
#    all_peak_values = all_peak_values.T

all_peak_values = peak_values_df
all_peak_values['BrIC'] = np.sqrt((all_peak_values['rv_x'].astype(float)/66.25)**2+(all_peak_values['rv_y'].astype(float)/56.45)**2+(all_peak_values['rv_z'].astype(float)/42.87)**2)
# 66.25, 56.45, 42.87
# √((ω_x/ω_xC )^2+(ω_y/ω_yC )^2+(ω_z/ω_zC )^2 )

#    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#        print(all_peak_values)

# Convert all numerical columns to float
#    for i in ['la_x', 'la_y', 'la_z', 'rv_x', 'rv_y', 'rv_z', 'la_r', 'rv_r', 'ra_x', 'ra_y', 'ra_z', 'ra_r']:
#        all_peak_values[i] = all_peak_values[i].astype(float)

grouped = all_peak_values.groupby('TestID')
means = grouped.mean()
stds = grouped.std()
cv = stds/means

for i in means, stds, cv:
    i['TestID'] = i.index

means['ID'] = 'Mean'
stds['ID'] = 'Standard Deviation'
cv['ID'] = 'Coefficient of Variance'

# Set the index of each dataframe in the list to 'ID'
for i, df in enumerate([all_peak_values,means,stds,cv]):
    df.index = df['ID']

# add mean and coefficient of variation columns to the original dataframe
all_peak_values = pd.concat([all_peak_values,means,cv])

# Dropping the time column
all_peak_values = all_peak_values[['ID', 'TestID', 'la_x', 'la_y', 'la_z', 'rv_x', 'rv_y', 'rv_z', 'la_r', 'rv_r', 'ra_x', 'ra_y', 'ra_z', 'ra_r', 'BrIC']]

all_peak_values.to_csv(path+ID+suffix+'PeakValues.csv', index=False)
# Calculate the coefficient

# Plotting
plt.style.use('ggplot')
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Helvetica"]
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

kinematics = ['la_', 'rv_', 'ra_']
kinematics_labels = ['Linear Acceleration', 'Rotational Velocity', 'Rotational Acceleration']

for j in np.arange(0,len(kinematics),1):
    for k in tests:
        print(k)
        # Set up the subplots
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        fig.subplots_adjust(hspace=0.3)
        # getting data to plot
        plotting_data = all_data.loc[all_data['TestID'] == k]

        ymin = plotting_data[kinematics[j]+'x'].min()
        if plotting_data[kinematics[j]+'y'].min() < ymin:
            ymin = plotting_data[kinematics[j]+'y'].min()
        if plotting_data[kinematics[j]+'z'].min() < ymin:
            ymin = plotting_data[kinematics[j]+'z'].min()

        # Minimums
        # print(plotting_data[kinematics[j]+'x'].min(), plotting_data[kinematics[j]+'y'].min(), plotting_data[kinematics[j]+'z'].min())

        plt.setp(axs, xlim= (-0.01, 0.04), ylim=(ymin*1.1, plotting_data[kinematics[j]+'r'].max()*1.1))

        for i in plotting_data['ID'].drop_duplicates().tolist():
            ID = i[11:]

            plotting_data_part = plotting_data.loc[plotting_data['ID'] == i]
            # Plot each component on a separate subplot
            axs[0, 0].plot(plotting_data_part['Time (s)'], plotting_data_part[kinematics[j]+'x'], label=ID)
            axs[0, 0].set_title(kinematics_labels[j]+' (x component)', fontsize=12, fontfamily='Arial')
            axs[0, 1].plot(plotting_data_part['Time (s)'], plotting_data_part[kinematics[j]+'y'], label=ID)
            axs[0, 1].set_title(kinematics_labels[j]+' (y component)', fontsize=12, fontfamily='Arial')
            axs[1, 0].plot(plotting_data_part['Time (s)'], plotting_data_part[kinematics[j]+'z'], label=ID)
            axs[1, 0].set_title(kinematics_labels[j]+' (z component)', fontsize=12, fontfamily='Arial')
            axs[1, 1].plot(plotting_data_part['Time (s)'], plotting_data_part[kinematics[j]+'r'], label=ID)
            axs[1, 1].set_title(kinematics_labels[j]+' (resultant)', fontsize=12, fontfamily='Arial')

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=[0.5, 0.00])
        fig.suptitle(k+' '+kinematics_labels[j], fontsize=14, fontfamily='Arial')
        if kinematics[j] == 'la_':
            plt.savefig(path+kinematics[j]+k+ID[:-3]+lin_acc_filter_str+'.jpg')
        elif kinematics[j] == 'rv_':
            plt.savefig(path+kinematics[j]+k+ID[:-3]+rot_vel_filter_str+'.jpg')
        elif kinematics[j] == 'ra_':
            plt.savefig(path+kinematics[j]+k+ID[:-3]+rot_processing_str+'.jpg')
        else:
            print(kinematics[j])
            print('Error! Not sure what kinematic channel data is and therefore nothing was saved.')
        # exit()

### NEED TO INVESTIGATE THE EFFECT OF THE FOLLOWING:
# Filtering data file, then differentiating
# Filtering data file, then 5-point moving average differentiation
# Differentiating raw data and then filtering
# Differentiating raw data with a 5-point moving average and then filtering
