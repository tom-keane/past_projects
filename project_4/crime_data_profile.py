# ---------------------------Tom Keane: 01788365------------------------------
"""
Profiling methodology: The crime_data package contained a timer wrapper based on datetime to provide feedback
of code timings throughout the pipeline. Therefore, it was easiest to implement profiling using this timer wrapper rather
than another more sophisticated built-in method such as timeit. This wrapper compares the datetime.now() at the start
and end of the function to provide a timing for the code.

The test data was constructed using repeated copies of the data-set from 12/2019. This was done to ensure that the number
of data points was increasing linearly with each additional data-set. The profiling function first  empties the
london directory. It then iterates through copying the data-set four times into four separate folders within the
london directory. This means on the first iteration there is 4 repeated data-sets, on the second there is 8 , etc.
This is done until there are 24 repeated data-sets. The profiling method was designed to create the test data-sets in
the london directory to allow the use of the original functions without needing to be re-written for profiling.
The code is run 10 times to try and reduce variation. Ideally the code would be run more times (~1000) however this
would take a few hours.
"""

import crime_data as cd
import numpy as np
import os
import shutil


def profiling(n, k, file_name):
    if os.path.isdir('london'):
        for treename in os.listdir('london'):
            file_path = os.path.join('london', treename)
            shutil.rmtree(file_path)
    else:
        os.mkdir('./london')
    total_times = np.zeros((3, k-1))
    original_name = file_name
    os.rename(original_name + ".csv", original_name + str(1) + ".csv")

    for i in range(1, k):
        for j in range(i*4 - 3, i*4 + 1):
            os.mkdir('./london' + "/" + str(j))
            shutil.copy2(original_name + str(j) + ".csv", './london' + "/" + str(j))
            os.rename(original_name + str(j) + ".csv", original_name + str(j+1) + ".csv")
        for w in range(0, n):
            total_times[0, i-1] += cd.timer(cd.collect_crime_data)/n
            total_times[1, i-1] += cd.timer(cd.create_london_db)/n
            total_times[2, i-1] += cd.timer(cd.run_crime_pipeline)/n
        print(i)
    os.rename(original_name + str(4*k-3) + ".csv", original_name + ".csv")
    temp = total_times[:, :-1]
    temp = np.hstack((np.zeros((3, 1)), temp))
    time_inc = total_times - temp

    return total_times, time_inc


time_total, time_increase = profiling(10, 7, "2019-12-metropolitan-street")
print(time_total)
print(time_increase)

"""
Avg total time (10 run average):

                     4 months    8 months    12 months   16 months   20 months   24 months
collect_crime_data   0.1665471   0.3155989   0.5001404   0.6560897   1.0753552   1.7818286
create_london_db     3.2494064   6.3926909   9.6212733   12.7769568  14.6244509  17.0216974
run_crime_pipeline   3.5334369   7.0916352   10.4922784  14.589997   16.889435   20.0333741



Increase in Avg time from 4 months additional data (10 run average):
 
                     4 months    8 months    12 months   16 months   20 months   24 months
collect_crime_data   0.1665471   0.1490518   0.1845415   0.1559493   0.4192655   0.7064734
create_london_db     3.2494064   3.1432845   3.2285824   3.1556835   1.8474941   2.3972465
run_crime_pipeline   3.5334369   3.5581983   3.4006432   4.0977186   2.299438    3.1439391

The time increase for each additional four repeated data-sets is approximately equal for run_crime_pipeline and
create_london_db. This means the code run length  increases linearly with an increase in data. There appears to be a 
large jump in increase for the collect_crime_data when there is 20+ months of datasets. This seemed unusual and due to
the collect_crime_data function being the shortest to run, I decided to profile that function alone using 100 runs for 
each number of data sets. This resulted in the following:

Increase in Avg time from 4 months additional data (100 run average):
 
                     4 months    8 months    12 months   16 months   20 months   24 months
collect_crime_data   0.2033396   0.30775578  0.25956006  0.36973705  0.38040206  0.37516313

We see a roughly linear trend as the increase is roughly constant. The large increase previously seen at 20+ months is 
not present. This shows the variation in the increase is due to a variation in the individual executions of the code. 
This individual variation also explains the variation seen in the create_london_db and run_crime_pipeline. This would 
be reduced by running the code for more iterations on each data-set as done for collect_crime_data.

This profiling was constructed on a real repeated data set to ensure the timing was representative of real-world scaling
of the data-set by 4 months.

In conclusion, all 3 functions scale linearly in time with the total number of data. Adding 4 months of data should add 
3-4 seconds to run time.
"""