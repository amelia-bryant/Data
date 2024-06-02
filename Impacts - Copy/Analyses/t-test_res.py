import numpy as np
from scipy.stats import ttest_ind

def perform_t_tests(data_pairs, alpha=0.05):

    for idx, (pair_name, name1, data1, name2, data2) in enumerate(data_pairs):
        t_stat, p_value = ttest_ind(data1, data2)
        
        print(f"Comparison for {pair_name}: {name1} vs {name2}")
        print(f"T-statistic: {t_stat}")
        print(f"P-value: {p_value}")
        
        if p_value < alpha:
            print(f"Since the p-value ({p_value}) is less than the significance level ({alpha}), we reject the null hypothesis.")
            print("Conclusion: There is a significant difference between the two sets of data.")
        else:
            print(f"Since the p-value ({p_value}) is greater than the significance level ({alpha}), we fail to reject the null hypothesis.")
            print("Conclusion: There is no significant difference between the two sets of data.")
        print('-' * 50)

# Example named data pairs
RA_Front = ('RA_Front', 'Relaxed', [1458.201598,1369.232905,2331.510782, 1052.79537,898.134404,902.2425513,1130.567054,1504.425057,1241.51856], 
                     'Clenched', [2250.09164, 3798.07457, 2913.540006, 1950.924028, 1204.505197, 1478.527789, 2523.94804, 1892.941876, 3197.449806])
RA_Side = ('RA_Side', 'Relaxed', [3400.384966, 2222.610996, 3400.384966, 1459.159637, 1217.216434, 1326.432355,1745.69402, 1808.102282, 1781.71551], 
                     'Clenched', [2964.627713, 4990.727514, 6225.697227, 1418.698537, 1241.064869, 971.1615632, 2564.925642, 2046.727697,1829.527073])


LA_Front = ('LA_Front', 'Relaxed', [34.3370151, 28.01026649, 40.1708855, 22.06320746, 19.31918008, 18.34243043, 27.93507232, 34.8037302, 31.20536179], 
                     'Clenched', [37.57204962, 33.44812823, 41.22174616, 22.52305733, 19.24229924, 19.32374764, 23.86932879, 25.52238179, 37.30912468])
LA_Side = ('LA_Side', 'Relaxed', [73.20210981, 30.97198314,73.20210981, 14.46245907, 10.69411922, 19.64935543, 22.72400576, 36.63909, 19.70875232], 
                     'Clenched', [33.86868909, 43.9668301,42.30242987, 33.99393156, 27.95876861, 21.57852389, 25.64640381, 47.32042381, 36.4957073])


RV_Front = ('RV_Front', 'Relaxed', [2.24990212, 3.289639447, 1.865306918, 1.400449645, 2.051100681, 2.135765609, 3.069530056, 2.26840978, 1.894603092], 
                     'Clenched', [3.105701338, 3.198381873, 2.487061402,1.437301606,2.251291822,2.880719133,2.106968773,2.436239154,3.014995017])
RV_Side = ('RV_Side', 'Relaxed', [3.168138571,2.784343161,3.168138571,2.761811885,2.729391998,2.450927567,2.888126584,3.240949003,2.124366749], 
                     'Clenched', [2.879829556,3.214487676,3.59129116,2.595536934,2.721273367,2.235066293,2.755673942,2.503692531,2.147275179])
# List of named data pairs
data_pairs = [RA_Front, RA_Side, LA_Front,LA_Side, RV_Front,RV_Side]

# Perform t-tests and report the conclusions for each pair of named data sets
perform_t_tests(data_pairs)