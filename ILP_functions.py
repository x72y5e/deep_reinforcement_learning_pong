import pandas as pd

def assess_clause(sub_df, opt, negatives, drop_list):

    #here we extract the feature, score and sign
    opt_feature = opt[0]
    value_of_feature = opt[1]

    #tells us if this feature predicts positive or negative outcome
    if opt[2] > 0: sign = 1
    else: sign = 0    
    
    for i in range(len(sub_df['c'])):
            
        #if sign is 0, a positive response on the feature in question should classify a negative (0)
        #feature is positive and class if positive; or feature 0 and class 0
            
        #inverse correlation
        if sign == 0:
            if sub_df[opt_feature][i] == value_of_feature and sub_df['c'][i] == 1:
                #print("negative match")
                negatives += 1
            
            # no correlation / feature is not present
            elif sub_df[opt_feature][i] != value_of_feature:
                #this example is not caught - drop from sub-df and add to drop list
                place_holder1 = sub_df.iloc[i:i+1, 0:]
                place_holder2 = [drop_list, place_holder1]
                drop_list = pd.concat(place_holder2)
                sub_df = sub_df.drop([i])
                
        # positive correlation
        elif sign == 1:
            if sub_df[opt_feature][i] == value_of_feature and sub_df['c'][i] == 0:
                #print("negative match")
                negatives += 1
                
            # no correlation / feature is not present
            elif sub_df[opt_feature][i] != value_of_feature:
                #this example is not caught - drop and add to drop list
                place_holder1 = sub_df.iloc[i:i+1, 0:]
                place_holder2 = [drop_list, place_holder1]
                drop_list = pd.concat(place_holder2)
                sub_df = sub_df.drop([i])
        
    return sub_df, negatives, drop_list

def select_feature(df):
    #which feature best classifies the data
    top_score = 0
    sub_score = 0
    feature_choice = None
    sub_feature_choice = None
    temp = df.drop(['c'], axis = 1)
    #print(df)
    for feature in temp.columns:
        #get each value in range for this feature
        for value in df[feature].unique():
            score = compare(feature, value, df)
            #print(score)
            if abs(score) > abs(sub_score):
                sub_score = score
                #store the score with the feature, indicating pos/neg
                sub_feature_choice = [feature, value, sub_score]
        if abs(sub_score) > abs(top_score):
                top_score = sub_score
                #store the score with the feature, indicating pos/neg
                feature_choice = sub_feature_choice
        
    #print("top score:", top_score)
    #print(feature_choice)
    return(feature_choice)

        
#this function checks how well the given feature classifies the data
def compare(feature, value, df):
    pos_classifications = 0
    neg_classifications = 0
    i = 0
    for item in df[feature]:
        
        #print(item, df['c'][i])
        if item == value and df['c'][i] == 1:
            pos_classifications += 1
        elif item == value and df['c'][i] == 0:
            neg_classifications += 1
        elif item == value and df['c'][i] == 1:
            neg_classifications += 1
        elif item == value and df['c'][i] == 0:
            pos_classifications += 1
        i += 1
    
    if pos_classifications == neg_classifications: return 1
    
    elif pos_classifications > neg_classifications and neg_classifications == 0: return pos_classifications
    
    elif pos_classifications > neg_classifications: return pos_classifications / neg_classifications
    
    elif pos_classifications < neg_classifications and pos_classifications == 0: return -1*neg_classifications
    
    else: return -1*(neg_classifications/pos_classifications)
    # note that this return negative score if correlation is 0 /1
