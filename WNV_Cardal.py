#################################################################################################
# Kaggle "West Nile Virus" competition
# April-June 2015
# Kaggle username: "Cardal" 
#################################################################################################

from sklearn import preprocessing

import pandas as pd
import numpy as np
import scipy
import datetime
import time
import os
import gc

MISSING_VALUE = -999

#################################################################################################
# I/O functions
#################################################################################################

def read_files(base_dir):
    '''
    Read the input files: train.csv, test.csv, sampleSubmission.csv
    '''
    train  = pd.read_csv('%s/train.csv' % base_dir)
    test   = pd.read_csv('%s/test.csv' % base_dir)
    sample = pd.read_csv('%s/sampleSubmission.csv' % base_dir)
    #weather = pd.read_csv('%s/weather.csv' % base_dir) # we don't use weather data!
    return train, test, sample

def create_dirs(dirs):
    '''
    Make sure the given directories exist. If not, create them
    '''
    for dir in dirs:
        if not os.path.exists(dir):
            print 'Creating directory %s' % dir
            os.mkdir(dir)


#################################################################################################
# Statistics functions
#################################################################################################

def array_count(arr):
    '''
    Count number of occurrences of each value in given array. Return dictionary with counts. 
    '''
    counts = {}
    for v in arr:
        counts[v] = counts.get(v, 0) + 1
    return counts


#################################################################################################
# Data processing functions
#################################################################################################

def process_date(train, test):
    '''
    Extract the year, month and day from the date
    '''
    # Functions to extract year, month and day from dataset
    def create_year(x):
        return int(x.split('-')[0])
    def create_month(x):
        return int(x.split('-')[1])
    def create_day(x):
        return int(x.split('-')[2])
    
    for ds in [train, test]:
        ds['year' ] = ds.Date.apply(create_year)
        ds['month'] = ds.Date.apply(create_month)
        ds['day'  ] = ds.Date.apply(create_day)

def convert_species(train, test):
    '''
    Convert the Species field into 4 attributes: IsPipiens, IsPipiensRestuans (gets 0.5 for Pipiens and for Restuans),
    IsRestuans, and IsOther (for all other species).
    '''
    for df in [train, test]:
                                                                                    # % Wnv  / total count in train data
#         df['IsErraticus'      ] = (df['Species']=='CULEX ERRATICUS')*1            #   0%   /    1
        df['IsPipiens'        ] = ((df['Species']=='CULEX PIPIENS'  )*1 +           # 8.9%   / 2699
                                   (df['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
        df['IsPipiensRestuans'] = ((df['Species']=='CULEX PIPIENS/RESTUANS')*1 +    # 5.5%   / 4752
                                   (df['Species']=='CULEX PIPIENS'  )*0 + (df['Species']=='CULEX RESTUANS'  )*0)
        df['IsRestuans'       ] = ((df['Species']=='CULEX RESTUANS'  )*1 +          # 1.8%   / 2740
                                   (df['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
#         df['IsSalinarius'     ] = (df['Species']=='CULEX SALINARIUS')*1           #   0%   /   86
#         df['IsTarsalis'       ] = (df['Species']=='CULEX TARSALIS'  )*1           #   0%   /    6
#         df['IsTerritans'      ] = (df['Species']=='CULEX TERRITANS' )*1           #   0%   /  222
        df['IsOther'          ] = (df['Species']!='CULEX PIPIENS')*(df['Species']!='CULEX PIPIENS/RESTUANS')*(df['Species']!='CULEX RESTUANS')

def convert_categorical(train, test, columns):
    lbl = preprocessing.LabelEncoder()
    for col in columns:
        lbl.fit(list(train[col].values) + list(test[col].values))
        train[col] = lbl.transform(train[col].values)
        test [col] = lbl.transform(test [col].values)


#################################################################################################
# Feature construction
#################################################################################################

def add_trap_bias(train, test, bias_factor=50.0):
    '''
    Compute the bias per trap and add it as a feature "TrapBias" to the train and test set.
    The bias of a trap is the fraction of rows (in that trap) that contain WnvPresent=1 divided 
    by the global ratio of WnvPresent=1. However, in order to take into account the statistical
    significance of the bias (i.e., if a trap has very few rows, we cannot be sure it's biased),
    we also consider the p-value of the number of WnvPresent in that trap (in the training data) 
    with respect to the total distribution of WnvPresent. The p-value is computed using the 
    hyper-geometric distribution (i.e., we check to what extent is WnvPresent 
    over/under-represented in the trap).
    '''
    logit = lambda p: np.log(p) - np.log(1 - p) # works well only for 0<p<1
    traps = np.unique(np.concatenate([train['Trap'], test['Trap']]))
    num_total, num_wnv = len(train), np.sum(train['WnvPresent'])
    ratio_wnv = (num_wnv + 0.05) / (num_total + 1.0)
    traps_bias = {}  
    for trap in traps:
        num, wnv = np.sum(train['Trap']==trap), np.sum(train[train['Trap']==trap]['WnvPresent'])
        if num==0:
            bias = 1.0
        else:
            ratio = (wnv + 0.05) / (num + 1.0)
            prob = np.min([scipy.stats.hypergeom.sf(wnv-1, num_total, num_wnv, num), scipy.stats.hypergeom.cdf(wnv, num_total, num_wnv, num)])
            ##bias = (ratio/ratio_wnv) ** ((1.0 - prob)**2)
            bias = (ratio/ratio_wnv) ** np.max([0.0, np.min([1.0, (logit(1.0-prob)/bias_factor)])])
    #         print 'Trap %3d: num %4d , wnv %4d -> prob = %.5f , ratio = %7.4f -> bias = %.4f' % (trap, num, wnv, prob, ratio/ratio_wnv, bias)
        traps_bias[trap] = bias
    
    train['TrapBias'] = np.array(map(lambda trap: traps_bias[trap], train['Trap']), dtype=np.float32)
    test ['TrapBias'] = np.array(map(lambda trap: traps_bias[trap], test ['Trap']), dtype=np.float32)

def add_features(train, test):
    '''
    Add two simple features to the train and test datasets:
    DaysSinceAug1 - number of days since Aug 1
    Norm - normal approximation of the distribution of WnvPresent along a year (based on training data)
    '''
    for df in [train, test]:
        # Number of days since Aug 1
        df['DaysSinceAug1'] = np.array([(datetime.datetime(y,m,d) - datetime.datetime(y,8,1)).total_seconds()/(60*60*24.0) 
                                        for y,m,d in zip(df.year,df.month,df.day)], dtype=np.float32)
        
        # Normal approximation of the distribution of WnvPresent along a year (based on training data)
        norm_mean, norm_std = 24.0, 18.0
        df['Norm'] = np.array([scipy.stats.norm.pdf((d-norm_mean)/norm_std) for d in df['DaysSinceAug1']])
            

#################################################################################################
# Multirow leakage
#################################################################################################

# Approximating the number of mosquitos per trap based on multirow count:
#[len([n for s,n in zip(sp_tr_date,train['NumMosquitos']) if counts[s]==c]) for c in range(30)]
#[mp.mean([n for s,n in zip(sp_tr_date,train['NumMosquitos']) if counts[s]==c]) for c in range(30)]
# Multirow count    # in train    mean # mosquitos
# ---------------   -----------   -----------------
# 1                 7720          7
# 2                 918           19.2 
# 3                 336           23
# 4                 184           27.9
# 5                 205           23.7
# 6                 126           28.8
# 7                 140           32
# 8                 120           30.5
# 9                 54            42.7 
# 10                60            33.6
# 11                66            20.8
# 12+               577           46.7
# --> mean # mosquitos = ~ 7*(1+3.2*np.log(1.0+np.log(n)))
# approximation to stats in training data of mean # of mosquitos per multirow count (see stats above)
num_mosq_per_entry_count = 7.0*(1.0 + 3.2*np.log(1.0+np.log(np.arange(100)))) 

def calc_mult_counts_prob(test, mult_entries_func):
    '''
    Compute probability factors based on multirow counts.
    test - the test dataset
    mult_entries_func - a function that takes as input the multirow count and returns the factor;
       the predictions are eventually multiplied by these factors, so mult_entries_func should
       typically return 1.0 for count=1, and >1.0 for count>1
    '''
    sp_tr_date_test = zip(test['Species'],test['Trap'],test['Address'],test['Date'])
    counts_test = array_count(sp_tr_date_test)
    return np.array([mult_entries_func(counts_test[s]) for s in sp_tr_date_test])

def count_nearby_mult(ds, dist_func, max_days=7, max_lat=0.1, max_lng=0.1, same_trap=False, same_species=False, use_self=True, ignore_count1=False):
    '''
    Estimate the number of mosquitos per row, based on the multirow count of the row itself and similar rows.
    The number of mosquitos for a specific row with a count of C is taken as num_mosq_per_entry_count[C] (see above).
    These estimates are weighted according to the given dist_func, so the final estimate for the number of mosquitos
    for a specific row is the weighted average of all the nearby rows (where "nearby" is defined by the input 
    parameters and dist_func). 
    ds - dataset (train or test)
    dist_func - the distance function for setting the weight of each row based on its similarity to the current row;
       takes as input: dist_days, dist_geo, species1, species2 
          where dist_days is the number of days between the two rows; dist_geo is the geographical distance;
          species1 is the species of the current row; species2 is the species of each of the other rows
    max_days - maximum number of days to consider
    max_lat, max_lng - maximum latitude/longitude difference to consider
    same_trap - whether to consider only rows with the same trap as the current row
    same_species - whether to consider only rows with the same species as the current row
    use_self - whether to include the current row in the set of rows we're comparing it to
    ignore_count1 - whether to ignore rows with multirow count of 1
    '''
    sp_tr_date = zip(ds['Species'],ds['Trap'],ds['Address'],ds['Date'])
    counts = array_count(sp_tr_date)

    cond_ignore1 = np.array([counts[s]>1 for s in sp_tr_date])
        
    # Find close traps/dates and count multiple rows
    nearby_mult = [] 
    for s,y,d,lat,lng,sp,tr in zip(sp_tr_date,ds['year'],ds['DaysSinceAug1'],ds['Latitude'],ds['Longitude'],ds['Species'],ds['Trap']): 
        cond = (ds['year']==y) * (ds['DaysSinceAug1']>(d-max_days-0.01)) * (ds['DaysSinceAug1']<(d+max_days+0.01))
        if max_lat is not None:
            cond = cond * (ds['Latitude' ]>(lat-max_lat-(1E-5))) * (ds['Latitude' ]<(lat+max_lat+(1E-5)))
        if max_lng is not None:
            cond = cond * (ds['Longitude']>(lng-max_lng-(1E-5))) * (ds['Longitude']<(lng+max_lng+(1E-5)))
        if same_trap:
            cond = cond * (ds['Trap']==tr)
        if same_species:
            cond = cond * (ds['Species']==sp)
        if not use_self:
            cond = cond * np.array([ss!=s for ss in sp_tr_date])
        if ignore_count1:
            cond = cond * cond_ignore1
        dsc = ds[cond]
        counts_c = np.array([counts[ss] for ss,cc in zip(sp_tr_date,cond) if cc])
        if len(counts_c)==0:
            nearby_mult.append(num_mosq_per_entry_count[1]) # use the minimum possible
        else:
            num_mosqs = np.take(num_mosq_per_entry_count, counts_c)
            weights = dist_func(np.abs(d-dsc['DaysSinceAug1']), np.sqrt((lat-dsc['Latitude'])**2 + (lng-dsc['Longitude'])**2), 
                                np.ones(len(dsc), dtype=np.int32)*sp, np.asarray(dsc['Species']))
            if np.sum(weights)<1e-5:
                nearby_mult.append(num_mosq_per_entry_count[1]) # use the minimum possible
            else:
                assert np.all(weights>=0) and np.sum(weights)>=1e-5
                weights = weights / counts_c # divide weight by counts because we use each entry 'counts' times
                nearby_mult.append(np.sum(num_mosqs * weights)/np.sum(weights))

    return np.array(nearby_mult, dtype=np.float32)
        
def calc_mult_combined_probs(test, mult_counts_probs_test, mult_nearby_probs_test, mult_nearby_probs_test2):
    '''
    Combine the multirow probabilities into a single array of factors with which to multiply the predictions.
    test - the test data
    mult_counts_probs_test - multirow probabilities based on the count of each row
    mult_nearby_probs_test, mult_nearby_probs_test2 - multirow probabilities based on the counts of each
        row and "nearby" rows (ie, rows from close dates and/or nearby traps)
    '''
    sp_tr_date_test = zip(test['Species'],test['Trap'],test['Address'],test['Date'])
    counts_test = array_count(sp_tr_date_test)

    # Parameters (slightly) optimized using LB feedback:
    species_fac_pow      = 0.3  
    mult1_pow, mult1_mul = 1.2, 50
    mult2_pow            = 0.5  
    counts_test_sp1 = [counts_test.get((1,s[1],s[2],s[3]),1) for s in sp_tr_date_test] # counts for species=1 (Pipiens)
    counts_test_sp2 = [counts_test.get((2,s[1],s[2],s[3]),1) for s in sp_tr_date_test] # counts for species=2 (PipiensRestuans)
    counts_test_sp3 = [counts_test.get((3,s[1],s[2],s[3]),1) for s in sp_tr_date_test] # counts for species=3 (Restuans)
    mult_combined_probs_test = []
    for mn,mn2,mc,s,cnt1,cnt2,cnt3 in zip(mult_nearby_probs_test,mult_nearby_probs_test2,mult_counts_probs_test,sp_tr_date_test,
                                          counts_test_sp1,counts_test_sp2,counts_test_sp3):
        if counts_test[s]==1:
            # Multirow count is 1 -> use nearby multirow counts of nearby rows
            if (s[0]==1 or s[0]==3) and (cnt2>1):
                # Species is Pipiens or Restuans -> take into account PipiensRestuans (for same date+trap) 
                fac = np.max([2.5,mult_entries_func(cnt2)])**species_fac_pow
            elif (s[0]==2) and (cnt1>1 or cnt3>1):
                # Species is PipiensRestuans -> take into account Pipiens and Restuans (for same date+trap) 
                fac = np.max([2.5,mult_entries_func(np.max([cnt1,cnt3]))])**species_fac_pow
            else:
                fac = 1.0
            mult_combined_probs_test.append(fac * np.min([2.0,1.0+((mn-1.0)**mult1_pow)*mult1_mul]) * (mn2**mult2_pow))
        else:
            # Multirow count is >1 -> use factor derived from the multirow count (but at least 2.5)
            mult_combined_probs_test.append(np.max([2.5,mc]))
    return mult_combined_probs_test


#################################################################################################
# Per-year and per-species coefficients
#################################################################################################

def get_year_coeffs():
    '''
    Get coefficient for each test year.
    Coefficients were initially set according to multirow counts, and further optimized using LB feedback.
    '''
    # Number of multiple rows per year:  2008: 430 , 2010: 423 , 2012: 728 , 2014: 581
    # Relative to average (=540.5)    :  2008: 0.8 , 2010: 0.78, 2012: 1.35, 2014: 1.07
    # Optimized using LB feedback:
    return {2008: 0.6 , 2010: 0.55, 2012: 3.6, 2014: 0.7 }

def apply_yearly_means(test, preds, coeffs):
    '''
    Update test predictions with yearly coefficients.
    '''
    return preds * np.array([coeffs[year] for year in np.asarray(test['year'])])

def get_pipiens_coeffs():
    '''
    Get coefficients for species=Pipiens (per year).
    Coefficients were initially set according to multirow counts, and further optimized using LB feedback.
    '''
    # Number pf multiple rows for Pipiens : 2008: 31   , 2010: 50   , 2012: 46   , 2014: 122
    # Relative to total multirows per year: 2008: 0.072, 2010: 0.118, 2012: 0.063, 2014: 0.210
    # Normalized w.r.t average (=0.116)   : 2008: 0.6  , 2010: 1.0  , 2012: 0.5  , 2014: 1.8
    # Optimized using LB feedback:
    return {2008: 0.5, 2010: 0.7, 2012: 0.1, 2014: 1.0}

def get_restuans_coeffs():
    '''
    Get coefficients for species=Restuans (per year).
    Coefficients were initially set according to multirow counts, and further optimized using LB feedback.
    '''
    # Number pf multiple rows for Restuans: 2008: 50   , 2010: 117  , 2012: 104  , 2014: 219
    # Relative to total multirows per year: 2008: 0.116, 2010: 0.277, 2012: 0.143, 2014: 0.377
    # Normalized w.r.t average (=0.228)   : 2008: 0.5  , 2010: 1.2  , 2012: 0.6  , 2014: 1.6
    # Optimized using LB feedback:
    return {2008: 0.8, 2010: 0.65, 2012: 0.7, 2014: 1.2}

def get_pipiens_restuans_coeffs():
    '''
    Get coefficients for species=PipiensRestuans (per year).
    Coefficients were initially set according to multirow counts, and further optimized using LB feedback.
    '''
    # Number pf multiple rows for PipiensRestuans: 2008: 347  , 2010: 232  , 2012: 578  , 2014: 236
    # Relative to total multirows per year       : 2008: 0.807, 2010: 0.548, 2012: 0.794, 2014: 0.406
    # Normalized w.r.t average (=0.639)          : 2008: 1.2  , 2010: 0.9  , 2012: 1.2  , 2014: 0.6
    # Optimized using LB feedback:
    return {2008: 0.9, 2010: 1.2, 2012: 1.0, 2014: 1.2 }

def get_other_coeffs():
    '''
    Get coefficients for species=Other (per year).
    Coefficients are very small since these mosquitos have no Wnv in the training data.
    '''
    # In train data there are no Wnv for species=Other, so we assume they typically don't carry the virus 
    # and therefore use very small coefficients  
    return {2008: 0.001, 2010: 0.001, 2012: 0.001, 2014: 0.001 }
    
def apply_species_coeffs(test, preds, coeffs, species):
    '''
    Update test predictions with species per-year coefficients.
    '''
    return np.array([v*((1.0-spc) + spc*coeffs[y]) for v,y,spc in zip(preds,np.array(test['year']),np.array(test[species]==1))])


#################################################################################################
# Date/location-based outbreaks during test years
#################################################################################################

def prp_outbreaks_daily_factors(outbreaks, years, pow=2):
    '''
    Prepare daily multipliers given an array of outbreaks.
    Each outbreak is a factor for a specific date. It influences the days before and after it,
    so that close days effectively receive the same factor; days farther away get a factor
    that is a weighted combination of two outbreaks - the weights are determined by the
    parameter 'pow'. 
    '''
    outbreaks_daily_factors, outbreaks_daily_weights = {}, {}
    for year in years:
        outbreaks_daily_factors[year], outbreaks_daily_weights[year] = {}, {}
        for month in [5,6,7,8,9,10]:
            outbreaks_daily_factors[year][month], outbreaks_daily_weights[year][month] = {}, {}
            for day in range(1, 32 if month in [5,7,8,10] else 31):
                outbreaks_daily_factors[year][month][day], outbreaks_daily_weights[year][month][day] = 0.0, 0.0
    for year,month,day,factor in outbreaks:
        outbreaks_daily_factors[year][month][day] += factor
        outbreaks_daily_weights[year][month][day] += 1.0 
        od = datetime.datetime(year, month, day)
        for month2 in range(month-1,month+2):
            if not outbreaks_daily_factors[year].has_key(month2): continue
            for day2 in outbreaks_daily_factors[year][month2].keys():
                if day==day2: continue
                od2 = datetime.datetime(year, month2, day2)
                days_diff = np.abs((od2 - od).total_seconds()/(60*60*24.0))
                if days_diff>9: continue
                outbreaks_daily_factors[year][month2][day2] += factor * (1.0/days_diff**pow)
                outbreaks_daily_weights[year][month2][day2] += 1.0/days_diff**pow
    for year in outbreaks_daily_factors.keys():
        for month in outbreaks_daily_factors[year].keys():
            for day in outbreaks_daily_factors[year][month].keys(): 
                if outbreaks_daily_weights[year][month][day]==0: continue
                outbreaks_daily_factors[year][month][day] = outbreaks_daily_factors[year][month][day] / outbreaks_daily_weights[year][month][day]
    return outbreaks_daily_factors

def get_outbreaks(test):
    '''
    Get coefficients for Wnv outbreaks in each test year. Each outbreak spans two weeks.
    Outbreaks coefficients were initially set according to multirow counts and Wnv distribution in train data, 
    and further optimized using LB feedback.
    '''
    # Wnv fraction in train data per two-weeks period (all train years combined):
    #         1/6:   0  , 16/6:    0 , 1/7: 0.001, 16/7: 0.02 , 1/8: 0.05 , 16/8: 0.13 , 1/9: 0.12 , 16/9: 0.05 , 1/10: 0.01
    # -> No Wnv around 1/6, 16/6 ; rare Wnv around 1/7, 1/10  
         
    # Multirow count in test data per two-weeks period:
    #  2008:  1/6:    0 , 16/6:    0 , 1/7:   24 , 16/7:   27 , 1/8:   85 , 16/8:   93 , 1/9:  118 , 16/9:   72 , 1/10:   11   
    #  2010:  1/6:    6 , 16/6:    7 , 1/7:   48 , 16/7:  159 , 1/8:   76 , 16/8:   44 , 1/9:   32 , 16/9:   33 , 1/10:   18
    #  2012:  1/6:    7 , 16/6:   19 , 1/7:   28 , 16/7:  295 , 1/8:  204 , 16/8:   80 , 1/9:   35 , 16/9:   51 , 1/10:    9
    #  2014:  1/6:   14 , 16/6:   72 , 1/7:   48 , 16/7:   57 , 1/8:  129 , 16/8:  142 , 1/9:   73 , 16/9:   33 , 1/10:   13
    # Normalized by total multirow count per year:
    #  2008:  1/6: 0    , 16/6: 0    , 1/7: 0.05 , 16/7: 0.06 , 1/8: 0.20 , 16/8: 0.22 , 1/9: 0.27 , 16/9: 0.18 , 1/10: 0.02   
    #  2010:  1/6: 0.01 , 16/6: 0.01 , 1/7: 0.11 , 16/7: 0.38 , 1/8: 0.18 , 16/8: 0.10 , 1/9: 0.07 , 16/9: 0.08 , 1/10: 0.04
    #  2012:  1/6: 0.01 , 16/6: 0.02 , 1/7: 0.04 , 16/7: 0.40 , 1/8: 0.28 , 16/8: 0.11 , 1/9: 0.05 , 16/9: 0.07 , 1/10: 0.01
    #  2014:  1/6: 0.02 , 16/6: 0.12 , 1/7: 0.08 , 16/7: 0.10 , 1/8: 0.22 , 16/8: 0.24 , 1/9: 0.13 , 16/9: 0.06 , 1/10: 0.02
    # After normalizing by mean ratio per period (across all 4 years):
    #  2008:  1/6:  0   , 16/6:  0   , 1/7: 0.8  , 16/7:  0.3 , 1/8:  0.9 , 16/8:  1.3 , 1/9:  2.1 , 16/9:  1.8 , 1/10:  1.0   
    #  2010:  1/6:  2.1 , 16/6:  0.4 , 1/7: 1.6  , 16/7:  1.6 , 1/8:  0.8 , 16/8:  0.6 , 1/9:  0.6 , 16/9:  0.8 , 1/10:  1.6
    #  2012:  1/6:  0.8 , 16/6:  0.6 , 1/7: 0.5  , 16/7:  1.7 , 1/8:  1.3 , 16/8:  0.6 , 1/9:  0.4 , 16/9:  0.8 , 1/10:  0.5
    #  2014:  1/6:  2.0 , 16/6:  2.9 , 1/7: 1.1  , 16/7:  0.4 , 1/8:  1.0 , 16/8:  1.4 , 1/9:  1.0 , 16/9:  0.6 , 1/10:  0.9
    # (+adjust the above according to Wnv fraction in train data) 
    
    # Optimized using LB feedback:    
    outbreaks_08 = [(2008, 6, 1, 0.01), (2008, 6, 16, 0.01), (2008, 7, 1, 0.01), (2008, 7, 16, 0.9), (2008, 8, 1, 1.0), (2008, 8, 16, 1.0), (2008, 9, 1, 1.0), (2008, 9, 16, 1.0), (2008, 10, 1, 0.01) ] 
    outbreaks_10 = [(2010, 6, 1, 0.01), (2010, 6, 16, 0.01), (2010, 7, 1, 0.01), (2010, 7, 16, 1.5), (2010, 8, 1, 1.2), (2010, 8, 16, 1.1), (2010, 9, 1, 1.0), (2010, 9, 16, 0.8), (2010, 10, 1, 0.01) ]
    outbreaks_12 = [(2012, 6, 1, 0.01), (2012, 6, 16, 0.01), (2012, 7, 1, 3.0 ), (2012, 7, 16, 2.5), (2012, 8, 1, 2.5), (2012, 8, 16, 0.5), (2012, 9, 1, 0.1), (2012, 9, 16, 0.1), (2012, 10, 1, 0.01) ]
    outbreaks_14 = [(2014, 6, 1, 0.01), (2014, 6, 16, 1.0 ), (2014, 7, 1, 1.0 ), (2014, 7, 16, 0.5), (2014, 8, 1, 1.0), (2014, 8, 16, 1.8), (2014, 9, 1, 1.8), (2014, 9, 16, 0.9), (2014, 10, 1, 0.1 ) ]
    
    return outbreaks_08 + outbreaks_10 + outbreaks_12 + outbreaks_14

def apply_outbreaks(test, preds, outbreaks):
    '''
    Update test predictions using outbreaks coefficients.
    '''
    test_years = np.unique(test['year'])
    outbreaks_daily_factors = prp_outbreaks_daily_factors(outbreaks, test_years, pow=10) 
    return np.array([pr*outbreaks_daily_factors[y][m][d] for pr,y,m,d in zip(preds,test['year'],test['month'],test['day'])], dtype=np.float32) 

def get_mini_outbreaks(test):
    '''
    Get coefficients for short Wnv outbreaks in July+Aug of test year 2012. A mini-outbreak spans 3 days.
    This period has the largest number of multirows (~500), and (probably) the highest fraction of Wnv.
    Mini-outbreaks coefficients were initially set according to multirow counts and further optimized using LB feedback.
    '''
    # Multirow count from mid July to beginning of August 2012:
    #    16-18/7: 0 , 19-21/7: 122 , 22-24/7: 58 , 25-27/7: 56 , 28-30/7: 1 , 31/7-2/8: 102 
    # -> Slightly decrease probabilities on 17/7, 29/7; increase on 20/7, 1/8

    # Optimized using LB feedback:    
    mini_outbreaks  = [ (2012, 7, 17, 0.8), (2012, 7, 20, 1.2), (2012, 7, 23, 0.8), (2012, 7, 26, 1.2), (2012, 7, 29, 0.8), (2012, 8,  1, 1.2)]
    return mini_outbreaks

def apply_mini_outbreaks(test, preds, mini_outbreaks):
    '''
    Update test predictions using mini-outbreaks coefficients.
    '''
    for year,month,day,factor in mini_outbreaks:
        od = datetime.datetime(year, month, day)
        preds = np.array([pr*(1.0 if np.abs((datetime.datetime(y,m,d) - od).total_seconds()/(60*60*24.0))>1.5 else factor) 
                          for pr,y,m,d in zip(preds,test['year'],test['month'],test['day'])], dtype=np.float32)
    return preds 

def get_geo_outbreaks(test):
    '''
    Get coefficients for Wnv occurrence on a geographical basis (per test year).
    We partition the traps based on their coordinates; more specifically, we look at Latitude-Longitude
    and separate the traps into three groups: those that are closest to the 83-percentile of Latitude-Longitude
    (ie, the traps that are in the north-west), those that are closest to the 50-percentile (center), and
    those that are closest to the 17-percentile (south-east).
    The geo_outbreaks gives the coefficient for each of these three groups (per year).
    Coefficients were slightly optimized using LB feedback (except for 2010).
    '''
    # Average WnvPresent in each Latitude-Longitude group in train data:
    #         group1  group2  group3
    # train:   0.07    0.04    0.05
    # Multirow count for traps in each Latitude-Longitude group: 
    # 2008:     307      39      84
    # 2010:     329      40      54
    # 2012:     570     103      55
    # 2014:     401      76     104
    # -> Both in train and test data there seems to be more Wnv in group1 (north-west)!
    # NOTE: Coefficients for 2010 weren't prepared or tested on LB
    geo_outbreaks = {2008: (1.2,1,0.8), 2012: (1.4,1,0.7), 2014: (1.2,1,0.8)} 
    return geo_outbreaks

def apply_geo_outbreaks(test, preds, geo_outbreaks):
    '''
    Update test predictions using geo-outbreaks coefficients.
    '''
    lml_perc = [np.percentile(test['Latitude']-test['Longitude'],perc) for perc in [83,50,17]]
    # Compute distance from each test sample to Latitude-Longitude percentiles
    test_geo_dists = []
    for lml_p in lml_perc:
        test_geo_dists.append(np.abs([(lml-lml_p) for lml in (test['Latitude']-test['Longitude'])]))
    # Update predictions
    dist_pow = 4
    for geo_year,geo_biases in geo_outbreaks.iteritems():
        preds = np.array([v*(np.average(geo_biases, weights=[1.0/((1e-3) + test_geo_dists[tg][i])**dist_pow for tg in range(3)]) if y==geo_year else 1.0) 
                          for i,(v,y) in enumerate(zip(preds,test['year']))])
    return preds

def get_trap_outbreaks(test):
    '''
    Get coeffecients for trap 147 (ORD Terminal 5, O'Hare International Airport) for each test year.
    This trap has the largest number of mosquitos in the train data, and the largest multirow count in
    the test data. Therefore, it has the highest influence on the predictions results.
    Trap outbreaks coefficients were initially set according to multirow counts and further optimized using LB feedback.
    '''
    # Multirow count in trap 147:                   2008: 159 , 2010: 278 , 2012: 236 , 2014: 207
    # Normalized by total multirow count per year:  2008: 0.37, 2010: 0.65, 2012: 0.32, 2014: 0.35
    # Normalized by average fraction (0.427)     :  2008: 0.9 , 2010: 1.5 , 2012: 0.8 , 2014: 0.8
    
    # Optimized using LB feedback:
    trap_outbreaks = [(147, {2008: 1, 2010: 1.2, 2012: 1.2, 2014: 0.7})]
    return trap_outbreaks

def apply_trap_outbreaks(test, preds, trap_outbreaks):
    '''
    Update test predictions using trap outbreaks coefficients.
    '''
    for trap,yearly_facs in trap_outbreaks:
        test_trap = test[test['Trap']==trap]
        preds = np.array([v*(yearly_facs[y] if tr==trap else 1.0) for v,y,tr in zip(preds,test['year'],test['Trap'])])
    return preds


# ================================================================================================================================
# Main
# ================================================================================================================================

if __name__ == "__main__":
    import sys
    import json

    # Check command-line arguments
    if len(sys.argv)>1:
        print 'Usage: "python WNV_Cardal.py"'
        exit()
    
    # Read params from SETTINGS.json
    with open('SETTINGS.json') as f:
        json_params = json.load(f)

    print '===> Kaggle West Nile Virus competition: Cardal solution'
    print '     %s' % ', '.join(['%s=%s'%(k,v) for k,v in sorted(json_params.iteritems())])
    
    input_dir      = json_params['INPUT_DIR']
    submission_dir = json_params['SUBMISSION_DIR']
    
    # Make sure all directories exist
    print '-> Making sure all directories exist'
    if not os.path.exists(input_dir):
        raise RuntimeError('Input directory (%s) does not exist' % input_dir)
    create_dirs([submission_dir])

    # Load input datasets
    print '-> Reading input files'
    train, test, sample = read_files(input_dir) 
    print '   There are %d,%d training,test samples' % (len(train), len(test))
    
    # Basic pre-processing and conversions
    print '-> Processing features'
    process_date(train, test) # Extract year, month and day from date
    convert_species(train, test) # Convert species into binary features
    convert_categorical(train, test, ['Species', 'Street', 'Trap']) # Convert categorical data to numbers
    assert np.all(train != MISSING_VALUE) and np.all(test != MISSING_VALUE)
    
    # Add various features
    print '-> Adding features'
    add_features(train, test)

    # Prepare predictions for test dataset    
    print '-> Preparing test predictions'
    
    # Use Normal distribution as baseline
    print 'Using Normal distribution as base predictor'
    test_pred = np.array(test['Norm'])   

    # Apply per-year and per-species coefficients
    print 'Applying yearly means'
    test_year_means = get_year_coeffs()
    test_pred = apply_yearly_means(test, test_pred, test_year_means)
    
    print 'Applying species yearly biases'
    pip_year_fac = get_pipiens_coeffs()
    res_year_fac = get_restuans_coeffs()
    prs_year_fac = get_pipiens_restuans_coeffs()
    otr_year_fac = get_other_coeffs()
    test_pred = apply_species_coeffs(test, test_pred, pip_year_fac, 'IsPipiens')
    test_pred = apply_species_coeffs(test, test_pred, res_year_fac, 'IsRestuans')
    test_pred = apply_species_coeffs(test, test_pred, prs_year_fac, 'IsPipiensRestuans')
    test_pred = apply_species_coeffs(test, test_pred, otr_year_fac, 'IsOther')

    # Apply date/location-based outbreaks 
    print 'Applying outbreaks'
    outbreaks = get_outbreaks(test)
    test_pred = apply_outbreaks(test, test_pred, outbreaks)
    
    print 'Applying mini-outbreaks'
    mini_outbreaks = get_mini_outbreaks(test)
    test_pred = apply_mini_outbreaks(test, test_pred, mini_outbreaks)
    
    print 'Applying geo-outbreaks'
    geo_outbreaks = get_geo_outbreaks(test)
    test_pred = apply_geo_outbreaks(test, test_pred, geo_outbreaks)
    
    print 'Applying trap outbreaks'
    trap_outbreaks = get_trap_outbreaks(test)
    test_pred = apply_trap_outbreaks(test, test_pred, trap_outbreaks)

    # Compute traps bias from train data, apply it to precitions
    print 'Computing trap bias'
    add_trap_bias(train, test, bias_factor=50.0)
    trap_bias_pow = 0.5
    print 'Applying trap bias'
    test_pred = test_pred * (test['TrapBias']**trap_bias_pow) 
    
    # Compute probabilities according to multirow counts      
    print 'Calculating multirow counts probabilities'
    mult_entries_coeff = 1.3 # optimized using LB feedback
    mult_entries_func = lambda c:  1.0 + mult_entries_coeff*np.log(c) 
    mult_counts_probs_test = calc_mult_counts_prob(test, mult_entries_func)
          
    # Count nearby multirow counts (=the counts of similar rows) 
    print 'Counting nearby multirow counts #1 (35 days, dist factor 100000)'
    mult_dist_func  = lambda dist_days, dist_geo, species1, species2: (1.0/(1.0+dist_days)) * (1.0/(1.0+100000*dist_geo))
    nearby_mult_test = count_nearby_mult(test, mult_dist_func, max_days=35, max_lat=None, max_lng=None, same_trap=False, same_species=True) 
    mult_nearby_probs_test = np.array([mult_entries_func(1.0+m/50.0)/mult_entries_func(1.0+num_mosq_per_entry_count[1]/50.0) for m in nearby_mult_test])
    assert np.min(mult_nearby_probs_test)>=1.0 and np.max(mult_nearby_probs_test)<mult_entries_func(2.0)

    print 'Counting nearby multirow counts #2 (14 days, same trap)'
    mult_dist_func2 = lambda dist_days, dist_geo, species1, species2: (1.0/(1.0+dist_days)**1.5) 
    nearby_mult_test2 = count_nearby_mult(test, mult_dist_func2, max_days=14, max_lat=None, max_lng=None, same_trap=True, same_species=True) 
    mult_nearby_probs_test2 = np.array([mult_entries_func(1.0+m/50.0)/mult_entries_func(1.0+num_mosq_per_entry_count[1]/50.0) for m in nearby_mult_test2])
    assert np.min(mult_nearby_probs_test2)>=1.0 and np.max(mult_nearby_probs_test2)<mult_entries_func(2.0)

    # Combine multirow counts probabilities
    print 'Combining multirow counts probabilities'
    mult_combined_probs_test = calc_mult_combined_probs(test, mult_counts_probs_test, mult_nearby_probs_test, mult_nearby_probs_test2)
    
    # Apply combined multirow counts probabilities to predictions
    print 'Applying combined multirow counts probabilities'
    test_pred = test_pred * mult_combined_probs_test
    
    # Normalize predictions so that they'll be between 0 and 1 (actually, we use 0-0.5, it doesn't matter)
    print 'Normalizing prediction probabilities'
    predictions = test_pred / (2.0*np.max(test_pred))
    assert np.all(predictions>=0) and np.all(predictions<=1.0)
        
    # Save results to submission file
    if True:
        sample['WnvPresent'] = predictions
        filename = '%s/wnv_cardal_final.csv' % (submission_dir)
        print '-> Saving results file to %s' % filename
        sample.to_csv(filename, index=False)
            
    print 'Done.'    
        
    
     