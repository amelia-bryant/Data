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
LA_Front = ('LA_Front', 'Relaxed', [7.685898103,
5.649917576,
11.80732769,
3.259274076,
1.533443525,
1.129426396,
1.75439697,
8.725714382,
7.034826862], 
                     'Clenched', [32.87837943,
14.75999855,
29.13084909,
7.136103782,
4.122703735,
17.3874206,
17.87281621,
18.40752451,
14.38634764])
LA_Side = ('LA_Side', 'Relaxed', [4.714856164,
2.779519465,
4.714856164,
0.643272925,
0.103057515,
3.257391149,
2.057525803,
1.664714569,
2.354192618], 
                     'Clenched', [4.796747377,
4.055833656,
4.958778764,
1.19811936,
0.028429494,
0.67712935,
1.308815324,
0.845794174,
1.259338396])


RV_Front = ('RV_Front', 'Relaxed', [1.938377149,
2.018026816,
1.648413478,
1.01218554,
1.886194564,
1.714719882,
2.604569944,
2.237417325,
1.729586302], 
                     'Clenched', [2.093409051,
2.753348845,
1.36895973,
1.293299819,
1.897769756,
2.743765167,
1.226455776,
2.386580789,
2.613044941])
RV_Side = ('RV_Side', 'Relaxed', [2.257621799,
1.445262118,
2.257621799,
1.719775323,
1.437666201,
1.163818623,
1.04473332,
2.459500384,
1.382790029], 
                     'Clenched', [2.919566327,
2.415112727,
3.092453703,
1.585306361,
1.409536978,
0.902755588,
1.944756841,
1.23797643,
1.431831501])

# Run test
data_pairs = [LA_Front,LA_Side, RV_Front,RV_Side]
perform_t_tests(data_pairs)