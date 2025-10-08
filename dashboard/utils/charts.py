import plotly.express as px

def feature_importance_chart(df):
    return px.bar(df.sort_values('importance', ascending=True),
                  x='importance', y='feature',
                  orientation='h',
                  color='importance',
                  color_continuous_scale='Blues',
                  title='🔍 Feature Importance – CTR Model')

def ctr_distribution(df):
    return px.histogram(df, x='predicted_prob',
                        nbins=30,
                        color='ad_format',
                        title='CTR Probability Distribution by Ad Format')

def roi_vs_ctr(df):
    return px.scatter(df, x='roi_percentage', y='ctr',
                      color='ad_format',
                      size='budget',
                      hover_name='advertiser',
                      title='📈 ROI vs CTR by Campaign')