import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

st.set_page_config(page_title='RaceLab',page_icon='🏇',layout='wide')
st.title('🏇 RaceLab | Racing prediction research')
st.caption('Historical backtesting • race-by-race probabilities • no fabricated racing data')
st.warning('Research prototype. Predictions are not guarantees. No live race feed is connected yet.')
with st.expander('Required CSV format'):
 st.code('race_id,race_date,horse,finish_position,field_size,odds_decimal,days_since_run,prior_runs,prior_wins,prior_places,prior_avg_finish,draw,distance_furlongs,going_code,trainer_30d_win_rate,jockey_30d_win_rate')
 st.write('Every feature must have been known before the race. Each race must contain all runners, including non-winners.')
file=st.file_uploader('Upload historical race results CSV',type='csv')
if file:
 df=pd.read_csv(file)
 required=['race_id','race_date','horse','finish_position','field_size','days_since_run','prior_runs','prior_wins','prior_places','prior_avg_finish','draw','distance_furlongs','going_code','trainer_30d_win_rate','jockey_30d_win_rate']
 missing=[x for x in required if x not in df.columns]
 if missing: st.error('Missing columns: '+', '.join(missing));st.stop()
 df['race_date']=pd.to_datetime(df['race_date'],errors='coerce')
 df=df.dropna(subset=['race_date','finish_position','race_id'])
 df['won']=(df.finish_position==1).astype(int)
 df['placed']=(df.finish_position<=df.field_size.map(lambda n: 2 if n<=4 else (3 if n<=7 else 4))).astype(int)
 features=['odds_decimal','days_since_run','prior_runs','prior_wins','prior_places','prior_avg_finish','draw','distance_furlongs','going_code','trainer_30d_win_rate','jockey_30d_win_rate']
 df['going_code']=df.going_code.astype(str)
 df=pd.get_dummies(df,columns=['going_code'],dtype=int)
 features=[c for c in df.columns if c in features or c.startswith('going_code_')]
 for c in features: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(df[c].median() if df[c].notna().any() else 0)
 dates=sorted(df.race_date.unique())
 if len(dates)<5 or df.race_id.nunique()<20: st.error('Need at least 5 racing dates and 20 races for this prototype.');st.stop()
 cutoff=dates[int(len(dates)*.8)]
 train=df[df.race_date<cutoff].copy();test=df[df.race_date>=cutoff].copy()
 if train.won.nunique()<2: st.error('Insufficient winners in training data');st.stop()
 models={}
 for target in ['won','placed']:
  model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=12,l2_regularization=5,random_state=42)
  model.fit(train[features],train[target]);test[target+'_prob']=model.predict_proba(test[features])[:,1];models[target]=model
 test['win_probability']=test.groupby('race_id')['won_prob'].transform(lambda x:x/x.sum() if x.sum()>0 else 1/len(x))
 selections=test.loc[test.groupby('race_id').win_probability.idxmax()].copy()
 a,b,c=st.columns(3)
 a.metric('Held-out races',len(selections));b.metric('Top-pick win rate',f'{selections.won.mean():.1%}');c.metric('Top-pick place rate',f'{selections.placed.mean():.1%}')
 st.caption('Chronological 80/20 split; results use unseen later races. Place rules are simplified for evaluation, not bookmaker settlement rules.')
 st.subheader('Historical out-of-sample selections')
 st.dataframe(selections[['race_date','race_id','horse','win_probability','placed_prob','finish_position','won','placed']].sort_values('race_date',ascending=False),hide_index=True,use_container_width=True)
 st.subheader('All test runners')
 st.dataframe(test[['race_date','race_id','horse','win_probability','placed_prob','finish_position']].sort_values(['race_date','race_id','win_probability'],ascending=[False,True,False]),hide_index=True,use_container_width=True)
 st.download_button('Download evaluated predictions',test.to_csv(index=False).encode(),'race_predictions.csv','text/csv')
