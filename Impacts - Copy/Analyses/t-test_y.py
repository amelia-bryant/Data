import numpy as np
from scipy.stats import ttest_ind

def perform_t_tests(data_pairs, alpha=0.05):

    for idx, (pair_name, name1, data1, name2, data2) in enumerate(data_pairs):
        t_stat, p_value = ttest_ind(data1, data2)
        
        print(f"Comparison for {pair_name}: {name1} vs {name2}")
        print(f"T-statistic: {t_stat}")
        print(f"P-value: {p_value}")
        
        if p_value < alpha:
            print(f"Since the p-value ({p_value}) is less than the significance level ({alpha}), reject the null.")
            print("Conclusion: There is a significant difference between the two sets of data.")
        else:
            print(f"Since the p-value ({p_value}) is greater than the significance level ({alpha}), fail to reject the null.")
            print("Conclusion: There is no significant difference between the two sets of data.")
        print('-' * 50)

# Data pairs
LA_Front = ('LA_Front', 'Relaxed', [8.881022027,
5.192951386,
8.635530801,
0.766835921,
1.977864857,
1.510214715,
5.835566294,
5.882561426,
5.266882996], 
                     'Clenched', [15.20187912,
7.426181742,
24.03039615,
4.3893775,
3.799167708,
7.594550339,
14.15996138,
15.31480286,
13.50846296])
LA_Side = ('LA_Side', 'Relaxed', [772.28148523,
34.18481105,
72.28148523,
16.12607385,
12.40677167,
20.95461492,
24.97092725,
35.4499669,
20.98330078], 
                     'Clenched', [31.16195285,
43.86180279,
43.05285723,
33.26971065,
28.74443064,
22.57949498,
24.68174808,
43.43104715,
35.86692416])


RV_Front = ('RV_Front', 'Relaxed', [1.275173293,
3.245204435,
1.783681921,
1.129969414,
1.383971202,
1.409507233,
1.621015169,
1.547456633,
1.464916869], 
                     'Clenched', [2.272805261,
2.05590625,
2.044432922,
1.450569075,
1.781948941,
1.679975703,
1.623581832,
1.36731795,
1.492644223])
RV_Side = ('RV_Side', 'Relaxed', [0.813086161,
0.633148498,
0.813086161,
0.387235228,
0.598830396,
0.905066266,
1.277921016,
0.750774309,
0.86099325], 
                     'Clenched', [0.743756626,
0.872132242,
1.370481001,
0.547917876,
0.527210219,
0.439113415,
0.716463545,
0.596433258,
0.511788255])

# Run test
data_pairs = [LA_Front,LA_Side, RV_Front,RV_Side]
perform_t_tests(data_pairs)