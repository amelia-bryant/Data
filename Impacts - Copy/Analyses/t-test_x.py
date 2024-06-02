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

LA_Front = ('LA_Front', 'Relaxed', [33.61099419,
27.8335907,
38.33468705,
22.07797216,
19.98320868,
19.11046429,
28.36888937,
34.13020956,
31.03213155], 
                     'Clenched', [11.06594568,
29.48384069,
16.75014265,
21.99842562,
18.80908989,
4.701378783,
7.572218395,
9.113811753,
32.37261175])
LA_Side = ('LA_Side', 'Relaxed', [19.11811122,
0.860039278,
19.11811122,
2.388991507,
1.01933202,
3.439774255,
0.061020019,
14.53716452,
2.556901175], 
                     'Clenched', [15.89478571,
10.51072032,
4.683340962,
12.19357974,
3.66391332,
0.676487583,
10.49136175,
19.42578976,
12.79040825])


RV_Front = ('RV_Front', 'Relaxed', [0.70723137,
1.023335917,
0.682658454,
0.796634035,
0.813691298,
0.941009911,
0.762227553,
0.687204145], 
                     'Clenched', [1.244250066,
0.824865944,
0.815017592,
0.731401902,
0.897160527,
0.965760405,
0.860904455,
0.820671661,
0.822252549])
RV_Side = ('RV_Side', 'Relaxed', [2.757295647,
2.68024375,
2.757295647,
2.651622551,
2.701501131,
2.295828395,
2.72390971,
3.038822401,
1.992546214], 
                     'Clenched', [2.798687156,
2.416435309,
2.851701763,
2.434158829,
2.368483004,
2.246300383,
2.678167368,
2.255291346,
2.216257517])
# List of named data pairs
data_pairs = [LA_Front,LA_Side, RV_Front,RV_Side]

# Perform t-tests and report the conclusions for each pair of named data sets
perform_t_tests(data_pairs)