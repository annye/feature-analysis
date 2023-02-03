import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

import pdb


class EDAworkflow(object):

    def __init__(self):
        ''' Reads dataset'''
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        # print('Version Of Python: ' + sys.version)
        # print('Version Of Pandas: ' + pd.__version__)
        # print('Version Of Numpy: ' + np.version.version)

        # Read data

        self.df = pd.read_csv(r'C:\Users\baffi\PycharmProjects\feature_selection\static\SURVEYDATA.csv')
        # Prints the number of columns and rows
        print('Total participants in survey:', len(self.df))
        print('Total survey items:', len(list(self.df.columns)))
        # self.df = pd.read_csv(data)

    def get_columns(self):
        '''Gets the column names for the data'''
        print(list(self.df.columns))

    def get_stats(self):
        '''Basic statistical info'''
        print(self.df.describe().T)

    def get_personality(self, df):
        """ Transforms, maps and calculates personality test reverse scores
                TIPI scale scoring (“R” denotes reverse-scored items):
                Extraversion: 1, 6R; Agreeableness: 2R, 7; Conscientiousness; 3, 8R; Emotional Stability: 4R, 9;
                Openness to Experiences: 5, 10R.
                Step 1. Recode the reverse-scored items
                (i.e., recode a 7 with a 1, a 6 with a 2, a 5 with a 3, etc.). 
                The reverse scored items are 2, 4, 6, 8, & 10.
                Step 2. Take the AVERAGE of the two items 
                (the standard item and the recoded reverse-scored item) that make up each scale.

            ----------
            x: ndarray
                The input samples - 10 n_features].

            Returns
            -------
            scores_and_labels: 10 features scores and  5 persoanlity dimension labels
            
            
            """

        def convert_To_float(row):
            try:
                return float(row)
            except:
                return None

        # Select tipi questions columns

        tipi = self.df[ [ 'Extraverted, enthusiastic',
                          'Critical, quarrelsome',
                          'Dependable, self-disciplined',
                          'Anxious, easily upset',
                          'Open to new experiences',
                          'Reserved, quiet',
                          'Sympathetic, warm',
                          'Disorganized, careless',
                          'Calm, emotionally stable',
                          'Conventional, uncreative' ] ]

        # Scoring key TIPI

        value = {'Disagree Strongly': '1',
                 'Disagree Moderately': '2',
                 'Disagree a Little': '3',
                 'Neither Agree nor Disagree': '4',
                 'Agree a Little': '5',
                 'Agree Moderately': '6',
                 'Agree Strongly': '7'}

        # Step 1. Mapping answers to numeric values TIPI
        for item in value:
            tipi = tipi.replace(value)

        # Step 2. TIPI Calculations: Reverse score =  (7 + 1 - [x]) --> 7 because the Likert scale max value is 7.

        # Extroversion
        reverse_score_1 = 7 + 1 - tipi[ 'Reserved, quiet' ].apply(convert_To_float)
        tipi[ 'Extroversion' ] = reverse_score_1 + tipi[ 'Extraverted, enthusiastic' ].apply(convert_To_float)
        tipi[ 'Extroversion' ] = tipi[ 'Extroversion' ] / 2

        # Agreeableness
        reverse_score_2 = 7 + 1 - tipi[ 'Sympathetic, warm' ].apply(convert_To_float)
        tipi[ 'Agreeableness' ] = reverse_score_2 + tipi[ 'Critical, quarrelsome' ].apply(convert_To_float)
        tipi[ 'Agreeableness' ] = tipi[ 'Agreeableness' ] / 2

        # Conscientiousness
        reverse_score_3 = 7 + 1 - tipi[ 'Disorganized, careless' ].apply(convert_To_float)
        tipi[ 'Conscientiousness' ] = reverse_score_3 + tipi[ 'Dependable, self-disciplined' ].apply(convert_To_float)
        tipi[ 'Conscientiousness' ] = tipi[ 'Conscientiousness' ] / 2

        # Emotional Stability
        reverse_score_4 = 7 + 1 - tipi[ 'Calm, emotionally stable' ].apply(convert_To_float)
        tipi[ 'Emotional stability' ] = reverse_score_4 + tipi[ 'Anxious, easily upset' ].apply(convert_To_float)
        tipi[ 'Emotional stability' ] = tipi[ 'Emotional stability' ] / 2

        # Openness
        reverse_score_5 = 7 + 1 - tipi[ 'Conventional, uncreative' ].apply(convert_To_float)
        tipi[ 'Openness' ] = reverse_score_5 + tipi[ 'Open to new experiences' ].apply(convert_To_float)
        tipi[ 'Openness' ] = tipi[ 'Openness' ] / 2

        # Calling DataFrame constructor on list
        tipi = pd.DataFrame(tipi)

        print('--------')
        print('TIPI survey section- shape:', tipi.shape)
        return tipi

    def get_belief_system(self, df):
        """ Calculate  DAS scores and labels for input data.
                
                Parameters
                ----------
                x: ndarray
                    The input samples [35 questions].

                Returns
                -------
                scores_and_labels: 35 questions scores + 7 das labels
                
                """

        # Subset das questions from survey data

        das = self.df.loc[ :, 'das1':'das35' ]

        # Key for mapping das scores

        das_scores = {'Agree Strongly': '-2',
                      'Agree Slightly': '-1',
                      'Neutral': '0',
                      'Disagree Slightly': '1',
                      'Disagree Very Much': '2'}

        for v in das_scores:
            das = das.replace(das_scores)

            # subsetting approval
            approval = das.loc[ :, 'das1': 'das5' ].astype(int)

            columns1 = list(approval.columns)
            das[ 'approval' ] = approval[ columns1 ].sum(axis=1)

            # subsetting Love
            love = das.loc[ :, 'das6': 'das10' ].astype(int)
            columns2 = list(love.columns)
            das[ 'love' ] = love[ columns2 ].sum(axis=1)

            # subsetting achievement
            achievement = das.loc[ :, 'das11': 'das15' ].astype(int)
            columns3 = list(achievement.columns)
            das[ 'achievement' ] = achievement[ columns3 ].sum(axis=1)

            # Subsetting perfectionism
            perfectionism = das.loc[ :, 'das16': 'das20' ].astype(int)
            columns4 = list(perfectionism.columns)
            das[ 'perfectionism' ] = perfectionism[ columns4 ].sum(axis=1)

            # Subsetting entitlement

            entitlement = das.loc[ :, 'das21': 'das25' ].astype(int)
            columns5 = list(entitlement.columns)
            das[ 'entitlement' ] = entitlement[ columns5 ].sum(axis=1)

            # subsetting omnipotence

            omnipotence = das.loc[ :, 'das26': 'das30' ].astype(int)
            columns6 = list(omnipotence.columns)
            das[ 'omnipotence' ] = omnipotence[ columns6 ].sum(axis=1)

            # subsetting autonomy

            autonomy = das.loc[ :, 'das31': 'das35' ].astype(int)
            columns7 = list(autonomy.columns)
            das[ 'autonomy' ] = autonomy[ columns7 ].sum(axis=1)

            # Calling DataFrame constructor on list
            das = pd.DataFrame(das)
            print('--------')
            print('DAS survey section- shape:', das.shape)
            return das

    def get_techniques(self, df):
        '''Persuasive statements'''
        techniques = self.df[ [
            'social-proof-1', 'social-proof-2', 'social-proof-3'
            , 'authority-1', 'authority-2', 'authority-3'
            , 'flattery-1', 'flattery-2', 'flattery-3'
            , 'rheto-question-1', 'rheto-question-2', 'rheto-question-3'
            , 'antanagoge-1', 'antanagoge-2', 'antanagogue-3'
            , 'logic-1', 'logic-2', 'logic-3'
            , 'pathos-1', 'pathos-2', 'pathos-3'
            , 'repetition-1', 'repetition-2', 'repetition-3'
            , 'anaphora-1', 'anaphora-2', 'anaphora-3'
            , 'main-argument-context-positive-framing-topic1'
            , 'main-argument-context-positive-framing-topic2'
            , 'main-argument-context-positive-framing-topic3'
        ] ]

        techniques.rename(columns={
            'main-argument-context-positive-framing-topic1': 't1_d1',
            'main-argument-context-positive-framing-topic2': 't1_d2',
            'main-argument-context-positive-framing-topic3': 't1_d3',

            'social-proof-1': 't2_d1',
            'social-proof-2': 't2_d2',
            'social-proof-3': 't2_d3',

            'flattery-1': 't3_d1',
            'flattery-2': 't3_d2',
            'flattery-3': 't3_d3',

            'rheto-question-1': 't4_d1',
            'rheto-question-2': 't4_d2',
            'rheto-question-3': 't4_d3',

            'antanagoge-1': 't5_d1',
            'antanagoge-2': 't5_d2',
            'antanagogue-3': 't5_d3',

            'logic-1': 't6_d1',
            'logic-2': 't6_d2',
            'logic-3': 't6_d3',

            'authority-1': 't7_d1',
            'authority-2': 't7_d2',
            'authority-3': 't7_d3',

            'pathos-1': 't8_d1',
            'pathos-2': 't8_d2',
            'pathos-3': 't8_d3',

            'repetition-1': 't9_d1',
            'repetition-2': 't9_d2',
            'repetition-3': 't9_d3',

            'anaphora-1': 't10_d1',
            'anaphora-2': 't10_d2',
            'anaphora-3': 't10_d3'
        }, inplace=True)

        techniques = pd.DataFrame(techniques)
        print('--------')
        print('Techniques survey section- shape:', techniques.shape)

        return techniques

    def get_demographics(self, df):

        demo = self.df[ [ 'age', 'gender', 'education' ] ]
        label_encoder = LabelEncoder()
        demo[ 'age' ] = label_encoder.fit_transform(demo[ 'age' ])
        demo[ 'gender' ] = label_encoder.fit_transform(demo[ 'gender' ])
        demo[ 'education' ] = label_encoder.fit_transform(demo[ 'education' ])

        # Calling DataFrame constructor on list
        demo = pd.DataFrame(demo)
        return demo
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    # class ML_worKflow(EDAworkflow):
#     """ Prepares datasetfor MLworkflow """

#     #  __init__ method of the EDAworkflow class
# #     def __init__(self):
# #         EDAworkflow.__init__(self)


#     def data_preparation(self):
#         """ Calling Object Data Survey"""

#         data= EDAworkflow()

#         # Tipi and das ready for analysis
#         tipi = data.get_personality(data)
#         das = data.get_belief_system(data)
#         demo = data.get_demographics(data)

#         # Influence scores

#         targets = data.get_techniques(data)


#         df = pd.concat([demo,tipi,das,targets],axis=1).astype(int)

#         # Selected input features
#         X = df[['age', 'gender', 'education', 'Extraverted, enthusiastic',
#             'Critical, quarrelsome', 'Dependable, self-disciplined',
#             'Anxious, easily upset', 'Open to new experiences', 'Reserved, quiet',
#             'Sympathetic, warm', 'Disorganized, careless',
#             'Calm, emotionally stable', 'Conventional, uncreative', 'das1',
#             'das2', 'das3', 'das4', 'das5', 'das6', 'das7', 'das8', 'das9',
#             'das10', 'das11', 'das12', 'das13', 'das14', 'das15', 'das16', 'das17',
#             'das18', 'das19', 'das20', 'das21', 'das22', 'das23', 'das24', 'das25',
#             'das26', 'das27', 'das28', 'das29', 'das30', 'das31', 'das32', 'das33',
#             'das34', 'das35']].astype(int)

#         print ("--Input Features shape:", X.shape)
#         print('------------')

#         return X, df, targets, data

#     def get_binary_targets(self,df, targets):
#             techniques = targets.columns
#             #Transforms the scores in a high/low scale"
#             for tech in techniques:
#                 criteria = [targets[tech].between(0, 5), targets[tech].between(6, 10)]
#                 values = [0, 1]
#                 targets[tech] = np.select(criteria, values) 
#                 targets.astype(int)


#         # techniques = targets.columns
#         # #Transforms the scores in a high/low scale"
#         # for tech in techniques:
#         #     criteria = [targets[tech].between(0, 5), targets[tech].between(6, 10)]
#         #     values = [0, 1]
#         #     targets[tech] = np.select(criteria, values) 
#         #     targets.astype(int)


#         #     #techniques,targets = binarise_tech(techniques)  

#         #     y = targets
#         #     print('------------')
#         #     print ("Targets shape:", y.shape)
#         #     print('------------')

#             # # split into train and test sets
#             # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,shuffle=True) 

#             # print ("Train feature shape:", X_train.shape)
#             # print('------------')
#             # print("Test feature shape:", X_test.shape)

#             return df, X, targets, techniques


#             #return df, X, targets, techniques, X_train, X_test, y_train, y_test


#     def get_scaler_sampling(self, X_train, X_test, y_train, y_test) :
#             """Scaling data before classifier."""

#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)

#             print ("Train scaled feature shape:", X_train.shape)
#             print("Test scaled feature shape:", X_test.shape)


#     # #          # define sampling strategy
#     #         under = RandomUnderSampler(random_state=42)

#     #         # fit and apply the transform
#             X_train, y_train = under.fit_resample(X_train, y_train)

#             return X_train, X_test, y_train, y_test


#####Calling the objects #####

#data = EDAworkflow(r'C:\Users\baffi\PycharmProjects\feature_selection\static\SURVEYDATA.csv')
# data.get_columns()
# data.get_dim()
# data.get_personality(data)
# data.get_belief_system(data)
# data.get_techniques(data)


################################


# Inizialization of object
# mlworkflow = ML_worKflow()
# #getting survey data ready to  ML modelling
# df, X, targets, techniques, X_train, X_test, y_train, y_test= mlworkflow.data_preparation()

# mlworkflow.get_scaler_sampling()
