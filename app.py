

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


st.title('MLB Next Pitch Predictor Tool')


st.write("""
Choose the in-game parameters to discover the predicted next pitch type!
""")
st.write('')
st.write("""
How to use:
Select your in-game situation (and pitcher) on the menu on the left and see the predicted next pitch of the at-bat!
""")


#Get data
df = pd.read_csv('mlb_2019_pitch_log.csv')
df = df.drop(['Unnamed: 0'], axis=1)




# for pitcher selection
pitcher = st.sidebar.selectbox(
        'Select Your Pitcher', 
        ('Hyun-Jin Ryu', 'Jacob deGrom', 'Kyle Hendricks', 
         'Lucas Giolito', 'Marcus Stroman', 'Trevor Bauer', 'Yu Darvish')
)

#pitcher = 'Lucas Giolito'      

# Add options for features
pitch_num = st.sidebar.slider('Pitch Number', 1, 120)

run_diff = st.sidebar.slider('Run Differential', 0, 10)

outs = st.sidebar.selectbox(
        'How Many Outs?', 
        ('0', '1', '2')
)

stand = st.sidebar.selectbox(
        "What side does the batter bat from?", 
        ('Right', 'Left'))

inning_half = st.sidebar.selectbox(
        'Top or Bottomg of the Inning?', 
        ('Top', 'Bottom'))

inning = st.sidebar.selectbox(
        'What inning is it?', 
        ('1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th'))

count = st.sidebar.selectbox(
        'What is the count?', 
        ('0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', 
         '3-1', '3-2'))


#get the correct pitcher
def get_pitcher(pitcher):
    p_id = 0
    
    if pitcher == 'Hyun-Jin Ryu':
        p_id = 547943
    if pitcher == 'Jacob deGrom':
        p_id = 594798
    if pitcher == 'Kyle Hendricks':
        p_id = 543294
    if pitcher == 'Lucas Giolito':
        p_id = 608337
    if pitcher == 'Marcus Stroman':
        p_id = 573186
    if pitcher == 'Trevor Bauer':
        p_id = 573186
    if pitcher == 'Yu Darvish':
        p_id = 506433
    
    return p_id


def get_outs(outs):
    outs_0 = 0
    outs_1 = 0
    outs_2 = 0
    
    if outs == '0':
        outs_0, outs_1, outs_2 = 1, 0, 0
    if outs == '1':
        outs_0, outs_1, outs_2 = 0, 1, 0
    if outs == '2':
        outs_0, outs_1, outs_2 = 0, 0, 1
    
    return outs_0, outs_1, outs_2


def get_stand(stand):
    stand_L = 0
    stand_R = 0
    
    if stand == 'Left':
        stand_L, stand_R = 1, 0
    if stand == 'Right':
        stand_L, stand_R = 0, 1
    
    return stand_L, stand_R


def get_inning_half(inning_half):
    inning_bottom = 0
    inning_top = 0
    
    if inning_half == 'Top':
        inning_bottom, inning_top = 0, 1
    if inning_half == 'Bottom':
        inning_bottom, inning_top = 1, 0
        
    return inning_bottom, inning_top


def get_inning(inning):
    inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    if inning == '1':
        inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9 = 1, 0, 0, 0, 0, 0, 0, 0, 0
    if inning == '2':
        inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9 = 0, 1, 0, 0, 0, 0, 0, 0, 0
    if inning == '3':
        inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9 = 0, 0, 1, 0, 0, 0, 0, 0, 0
    if inning == '4':
        inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9 = 0, 0, 0, 1, 0, 0, 0, 0, 0
    if inning == '5':
        inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9 = 0, 0, 0, 0, 1, 0, 0, 0, 0
    if inning == '6':
        inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9 = 0, 0, 0, 0, 0, 1, 0, 0, 0
    if inning == '7':
        inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9 = 0, 0, 0, 0, 0, 0, 1, 0, 0
    if inning == '8':
        inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9 = 0, 0, 0, 0, 0, 0, 0, 1, 0
    if inning == '9':
        inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9 = 0, 0, 0, 0, 0, 0, 0, 0, 1
        
    return inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9
    
    
######## THIS IS WHERE YOU LEFT OFF - NEED TO DO COUNT!
def get_count(count):
    count_0_0, count_0_1, count_0_2, count_1_0, count_1_1, count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 
    
    if count == '0-0':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1, count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2= 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if count == '0-1':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1,count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2= 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if count == '0-2':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1,count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0    
    if count == '1-0':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1, count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
    if count == '1-1':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1,count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
    if count == '1-2':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1,count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
    if count == '2-0':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1,count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
    if count == '2-1':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1,count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0
    if count == '2-2':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1,count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    if count == '3-0':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1,count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0
    if count == '3-1':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1,count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
    if count == '3-2':
        count_0_0, count_0_1, count_0_2, count_1_0, count_1_1,count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
           
    return (count_0_0, count_0_1, count_0_2, count_1_0, count_1_1, count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2)


#Get choices in variables for custom prediction
    
pitcher_id = get_pitcher(pitcher)

outs_0, outs_1, outs_2 = get_outs(outs)

stand_L, stand_R = get_stand(stand)

inning_bottom, inning_top = get_inning_half(inning_half)

inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9 = get_inning(inning)

count_0_0, count_0_1, count_0_2, count_1_0, count_1_1, count_1_2, count_2_0, count_2_1, count_2_2, count_3_0, count_3_1, count_3_2 = get_count(count)




# We will do the model twice. Once for general results
df = df.loc[df['pitcher_id'] == pitcher_id]



pt = df.pitch_type.unique()
nextP = df.next_pitch.unique()

pt = set(pt)
intersection = list(pt.intersection(nextP))
#print(intersection)


df = df[df['next_pitch'].isin(intersection)]
#df.next_pitch.value_counts()





x = df.drop(['next_pitch', 'pitcher_id', 'pitch_type'], axis = 1)


x = pd.get_dummies(x, columns=['outs', 'stand', 'top', 'inning', 'count'])


y = df['next_pitch']



#The model!
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 92 )

clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', max_depth=10).fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)

st.subheader('Pitcher Chosen:')
st.write('Pitcher: ', pitcher)
st.write('')

st.subheader('Overall Model Accuracy:')
st.write('Accuracy (%): ', (round(acc*100, 2)))
st.write('')


#Specific prediction asked
prediction = clf.predict([[pitch_num, run_diff, outs_0, outs_1, outs_2, stand_L, stand_R, inning_bottom, inning_top, 
                           inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9, 
                           count_0_0, count_0_1, count_0_2, count_1_0, count_1_1, count_1_2, count_2_0, count_2_1, 
                           count_2_2, count_3_0, count_3_1, count_3_2]])
    
prediction_prob = clf.predict_proba([[pitch_num, run_diff, outs_0, outs_1, outs_2, stand_L, stand_R, inning_bottom, inning_top, 
                           inning_1, inning_2, inning_3, inning_4, inning_5, inning_6, inning_7, inning_8, inning_9, 
                           count_0_0, count_0_1, count_0_2, count_1_0, count_1_1, count_1_2, count_2_0, count_2_1, 
                           count_2_2, count_3_0, count_3_1, count_3_2]])
classes = clf.classes_ 


st.subheader('Next Pitch of At-Bat')  
st.write('The predicted next pitch is: ', prediction)
st.write('')

st.subheader('Pitch Types')
st.write(classes)
st.write('')

st.subheader('Prediction Probability')
st.write('Use the Pitch Type index above to find the corresponding pitches. This is the likelihood of the pitch types to be thrown next in at-bat.')
st.write(prediction_prob)

st.text('')
st.text('')

st.subheader('Pitch Type Dictionary')
st.write("""
CH - Changeup

CU - Curveball

EP - Eephus*

FC - Cutter

FF - Four-seam Fastball

FO - Pitchout (also PO)*

FS - Splitter

FT - Two-seam Fastball

IN - Intentional ball

KC - Knuckle curve

KN - Knuckeball

PO - Pitchout (also FO)*

SC - Screwball*

SI - Sinker

SL - Slider
""")












