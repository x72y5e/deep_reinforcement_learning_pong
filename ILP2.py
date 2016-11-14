import pandas as pd
import ILP_functions as fs

df = pd.read_csv('iris.csv', header=0)

#this is the list of features
features = df.columns.values.tolist()[1:]

#this is the list for the output classification rules
rules = []
#these counters can be used to stop the algorithm looping for too long
count = 0
outer_count = 0

#outer loop: while positive examples remain to be classified
while len(df['c'])>0 and outer_count < 10:
    print("outer loop - building rule")    
    
    #set negative count to 1 to enter negatives loop
    negatives = 1
    
    #this is the clause to be built
    clause = []
    
    #duplicate dataframe for clause-builder loop
    sub_df = df
    
    #inner loop: build a rule until it describes only positives examples
    while negatives > 0 and count < 10:
        print("inner loop - building clause")
        
        #once in the loop, set negatives back to 0 to see if any are found
        negatives = 0
        
        #here is the dropped list - everything added must be dealt with by the
        #next clause
        drop_list = pd.DataFrame()
    
        opt_feature_and_score = fs.select_feature(sub_df)
        #this returns [feature, value, +/- score]
        
        clause.append(opt_feature_and_score)
        
        sub_df, negatives, drop_list = fs.assess_clause(sub_df, opt_feature_and_score, negatives, drop_list)

        print("negatives classified:", negatives)
        
        #reset index of temp df
        sub_df.index = range(len(sub_df['c']))
        
        count += 1
    
    #once all negative examples are excluded, add this clause to the rule, and continue
    #until all examples have been classified
    rules.append(clause)
    
    #remove classified examples from df
    if drop_list.empty != False:
        df = drop_list
        df.index = range(len(df['c']))
        #print(drop_list)
    
        outer_count += 1
    
    else: break

print(rules)
        
    
