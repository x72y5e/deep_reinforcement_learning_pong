import pandas as pd
import pylab as p
import numpy as np
import csv
import ILP_functions as fs

df = pd.read_csv('train-smpl.csv', header=0)

#this is the list of features
features = df.columns.values.tolist()[1:]
df['remaining'] = 0

#this is the list for the output classification rules
rules = []
#these counters can be used to stop the algorithm looping for too long
count = 0
outer_count = 0

#outer loop: while positive examples remain to be classified
while len(df['c'])>0 and outer_count < 10:
    print("while positives != []")
    
    #set negative count to 1 to enter negatives loop
    negatives = 1
    #this is the clause to be built
    clause = []
    
    #duplicate dataframe for clause-builder loop
    temp_df = df
    
    #inner loop: build a rule until it describes only positives examples
    while negatives > 0 and count < 10:
        print("while negatives > 0")
        
        #once in the loop, set negatives back to 0 to see if any are found
        negatives = 0
        
        #here is the dropped list - these must be dealt with by the next clause
        drop_list = pd.DataFrame()
    
        best_feature_and_score = fs.select_feature(temp_df)
        #this returns [feature, value, +/- score]
        #print(best_feature_and_score)
    
        clause.append(best_feature_and_score)
        best_feature = best_feature_and_score[0]
        value_of_feature = best_feature_and_score[1]
        print(best_feature_and_score)

        #tells us if this feature predicts positive or negative outcome
        if best_feature_and_score[2] > 0: sign = 1
        else: sign = 0
        #print(sign)
        
        
        #this loop determines how the clause classifies the 
        for i in range(len(temp_df['c'])):
            
            #if sign is 0, a positive response on the feature in question should classify a negative (0)
            #feature is positive and class if positive; or feature 0 and class 0
            
            #inverse correlation
            if sign == 0:
                if temp_df[best_feature][i] == value_of_feature and temp_df['c'][i] == 0:
                    print("positive match")
                    
                elif temp_df[best_feature][i] == value_of_feature and temp_df['c'][i] == 1:
                    print("negative match")
                    negatives += 1
                
                # no correlation / feature is not present
                elif temp_df[best_feature][i] != value_of_feature:
                    #this example is not caught - drop and add to drop list
                    place_holder1 = temp_df.iloc[i:i+1, 0:]
                    place_holder2 = [drop_list, place_holder1]
                    drop_list = pd.concat(place_holder2)
                    temp_df = temp_df.drop([i])
                    
            
            # positive correlation
            elif sign == 1:
                if temp_df[best_feature][i] == value_of_feature and temp_df['c'][i] == 1:
                    print("positive match")
                    
                elif temp_df[best_feature][i] == value_of_feature and temp_df['c'][i] == 0:
                    print("negative match")
                    negatives += 1
                
                # no correlation / feature is not present
                elif temp_df[best_feature][i] != value_of_feature:
                    #this example is not caught - drop and add to drop list
                    place_holder1 = temp_df.iloc[i:i+1, 0:]
                    place_holder2 = [drop_list, place_holder1]
                    drop_list = pd.concat(place_holder2)
                    temp_df = temp_df.drop([i])
        
        #now out of the deletion / comparison loop. If negatives found, repeat   
        #with refined to_classify data only including examples caught by the clause so far    
        print("negatives classified:", negatives)
        
        #reset index of temp df
        temp_df.index = range(len(temp_df['c']))
        
        count += 1
    
    #remove all 0 remains
    
    
    rules.append(clause)
    
    #remove classified examples from df
    if drop_list.empty != False:
        df = drop_list
        df.index = range(len(df['c']))
        print(drop_list)
    
        outer_count += 1
    
    else: break

print(df.describe())
print(rules)
        
    
