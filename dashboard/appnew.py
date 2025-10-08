# app.py — Netflix × Amazon DSP • Executive Intelligence Hub (dark theme)
# Works with your schema:
# campaigns, impressions, clicks, conversions, mrt_campaign_summary,
# ctr_predictions, ctr_feature_importance, user_segments

import os
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# ------------------ App + Theme ------------------
st.set_page_config(page_title="Netflix × Amazon DSP – Executive Intelligence Hub",
                   layout="wide", initial_sidebar_state="expanded")

# Netflix dark look & feel
st.markdown("""
<style>
html, body, [class*="css"]  {
  background-color: #0E0E0E !important;
  color: #FFFFFF !important;
}
.sidebar .sidebar-content { background-color:#0E0E0E; }
section.main > div { padding-top: 10px; }

.card { background:#181818;border:1px solid #2A2A2A;border-radius:14px;padding:18px 20px;box-shadow:0 0 12px rgba(229,9,20,0.10); }
.kpi { background:#181818;border:1px solid #2A2A2A;border-radius:14px;padding:16px 18px;box-shadow:0 0 8px rgba(229,9,20,0.08); }
.kpi .title { color:#B3B3B3;font-size:12px;letter-spacing:.25px;text-transform:uppercase;font-weight:600; }
.kpi .value { color:#FFFFFF;font-size:28px;font-weight:800;margin-top:2px; }
.kpi .sub   { color:#B3B3B3;font-size:12px;margin-top:2px; }

.h1 { font-size:30px;font-weight:900;color:#fff; }
.h2 { font-size:20px;font-weight:800;color:#fff; margin-bottom:2px;}
.caption { color:#B3B3B3;font-size:13px;margin:4px 0 12px 0;}
.hr { border:none; border-top:1px solid #2A2A2A; margin:10px 0 16px 0; }

.tag { display:inline-block; background:#251112; color:#FF4D57; border:1px solid #3A0D10;
       padding:2px 8px;border-radius:999px;font-size:11px; font-weight:700; letter-spacing:.3px; }
.red  { color:#E50914; }
.green{ color:#1DB954; }
.small{ font-size:12px;color:#B3B3B3; }

.info-box {
  background-color: #181818;
  border: 1px solid #E50914;
  border-radius: 12px;
  padding: 16px 22px;
  margin: 12px 0 20px 0;
  box-shadow: 0 0 8px rgba(229,9,20,0.15);
}
.info-box .title {
  font-weight: 800;
  color: #E50914;
  font-size: 16px;
  margin-bottom: 6px;
}
.info-box .body {
  color: #EDEDED;
  font-size: 16px;
  line-height: 1.6;
}

table { color:#EDEDED; }

</style>

""", unsafe_allow_html=True)

# ------------------ DB Connection ------------------
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5433")
DB_NAME = os.getenv("PGDATABASE", "ads_db")
DB_USER = os.getenv("PGUSER", "admin")
DB_PASS = os.getenv("PGPASSWORD", "admin")

DB_URL = "postgresql+psycopg://neondb_owner:npg_6OS3wVMzaFjN@ep-weathered-bird-adpzpean-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DB_URL, pool_pre_ping=True)

# engine = create_engine(
#     f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
#     pool_pre_ping=True
# )

@st.cache_data(ttl=300, show_spinner=False)
def load_df(sql: str) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn)
    
# ------------------ Load Required Tables ------------------
mrt = load_df("""
    select campaign_id, advertiser, target_region, ad_format,
           impressions_count, clicks_count, conversions_count,
           total_revenue::float as total_revenue, budget::float as budget,
           ctr::float as ctr, cvr::float as cvr, roi_percentage::float as roi_percentage,
           cpc::float as cpc, cpa::float as cpa, cpm::float as cpm,
           rpc::float as rpc, rpcv::float as rpcv, roas::float as roas,
           active_days, revenue_per_day::float as revenue_per_day
    from mrt_campaign_summary
""")

campaigns = load_df("""
    select campaign_id, advertiser, budget::float as budget, start_date, end_date
    from campaigns
""")

impr_daily = load_df("""
    select date_trunc('day', timestamp)::date as day, count(*)::bigint as impressions
    from impressions group by 1 order by 1
""")
click_daily = load_df("""
    select date_trunc('day', timestamp)::date as day, count(*)::bigint as clicks
    from clicks group by 1 order by 1
""")
rev_daily = load_df("""
    select date_trunc('day', timestamp)::date as day, sum(coalesce(revenue,0))::float as revenue
    from conversions group by 1 order by 1
""")

pred = load_df("""
    select region, device, subscription_tier, ad_format, target_region,
           budget::float as budget, hour_of_day::float as hour_of_day, day_of_week::float as day_of_week,
           is_peak_hour::int, user_click_rate::float, campaign_ctr_history::float,
           actual_clicked::int, predicted_prob::float, predicted_clicked::int
    from ctr_predictions
""")

feat = load_df("select feature, importance::float from ctr_feature_importance order by importance desc")

segments = load_df("""
    select user_id, region, device, subscription_tier,
           impressions::bigint, clicks::bigint, conversions::bigint,
           total_revenue::float as total_revenue, avg_ctr_score::float as avg_ctr_score,
           days_since_signup::int, user_cluster::int
    from user_segments
""")

# ------------------ Sidebar Filters ------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
                 use_column_width=True)
st.sidebar.markdown("<span class='tag'>Executive Intelligence Hub</span>", unsafe_allow_html=True)
st.sidebar.write("")

reg_f  = st.sidebar.multiselect("Filter • Region", sorted(mrt["target_region"].dropna().unique().tolist()))
fmt_f  = st.sidebar.multiselect("Filter • Ad Format", sorted(mrt["ad_format"].dropna().unique().tolist()))
adv_f  = st.sidebar.multiselect("Filter • Advertiser", sorted(mrt["advertiser"].dropna().unique().tolist()))

mask = pd.Series(True, index=mrt.index)
if reg_f: mask &= mrt["target_region"].isin(reg_f)
if fmt_f: mask &= mrt["ad_format"].isin(fmt_f)
if adv_f: mask &= mrt["advertiser"].isin(adv_f)
mrt_f = mrt.loc[mask].copy()

# Numeric helpers (robust to NaN/inf)
def ssum(s): return float(pd.to_numeric(s, errors="coerce").fillna(0).sum())
def smean(s):
    x = pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
    return float(x.mean()) if len(x) else 0.0

# KPI calculations
total_impr   = ssum(mrt_f["impressions_count"])
total_clicks = ssum(mrt_f["clicks_count"])
total_rev    = ssum(mrt_f["total_revenue"])
total_cost   = ssum(mrt_f["budget"])
avg_ctr      = smean(mrt_f["ctr"])
avg_cvr      = smean(mrt_f["cvr"])
avg_cpc      = smean(mrt_f["cpc"])
avg_cpm      = smean(mrt_f["cpm"])
avg_roi      = smean(mrt_f["roi_percentage"])
avg_roas     = smean(mrt_f["roas"])

# ------------------ Page Switcher ------------------
page = st.sidebar.radio(
    "Pages",
    ["1) Executive Summary",
     "2) Campaign Intelligence",
     "3) Predictive & Audience Insights"],
    index=0
)

# ======================================================
# 1) EXECUTIVE SUMMARY
# ======================================================
if page.startswith("1"):
    st.markdown("<div class='h1'> </div>", unsafe_allow_html=True)
    st.markdown("<div class='h1'>🎬 Netflix × Amazon DSP - Business Command Center</div>", unsafe_allow_html=True)
    st.markdown("<div class='caption'>Turning viewers into value. At-a-glance health across reach, engagement and efficiency.</div>", unsafe_allow_html=True)
    
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, title, val, sub in [
        (c1, "Impressions", f"{int(total_impr):,}", "Awareness reached"),
        (c2, "Clicks", f"{int(total_clicks):,}", "Engagement actions"),
        (c3, "Ad Revenue", f"${total_rev:,.0f}", "Attributed conversions"),
        (c4, "Ad Spend", f"${total_cost:,.0f}", "Planned budget burn"),
        (c5, "CTR", f"{(avg_ctr*100):.2f}%", "Ad appeal"),
        (c6, "ROI %", f"{avg_roi:,.2f}%", "Efficiency of spend"),
    ]:
        with col:
            st.markdown(f"<div class='kpi'><div class='title'>{title}</div>"
                        f"<div class='value'>{val}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)
    #st.markdown("<div class='h3'>Summary: Netflix ads through Amazon DSP reached 652 K viewers, generated 30 K clicks, and converted to $1.7 M in revenue on $1.3 M spend, achieving 29% ROI and a solid 4.5% CTR.</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-box'>
        <div class='title'>Insight</div>
        <div class='body'>
            A pulse check of campaign reach, engagement, revenue and efficiency for Netflix ads bought via Amazon DSP. <br>
            We’re profitable (<b>~30% ROI</b>) with healthy engagement (<b>~4.5% CTR</b>). The current media mix is converting attention into revenue efficiently.
        </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        ## st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>Where do dollars come from?</div>", unsafe_allow_html=True)
        st.markdown("<div class='caption'>Total <b>Ad Revenue</b> vs <b>Spend</b> by advertiser — bubble size = impressions.</div>", unsafe_allow_html=True)
        
        if not mrt_f.empty:
            df = (mrt_f.groupby("advertiser", as_index=False)
                  .agg(revenue=("total_revenue","sum"),
                       spend=("budget","sum"),
                       impressions=("impressions_count","sum")))
            fig = px.scatter(df, x="spend", y="revenue", size="impressions",
                             hover_name="advertiser", color="revenue",
                             color_continuous_scale="Reds",
                             labels={"spend":"Ad Spend","revenue":"Ad Revenue"})
            fig.update_layout(height=420, paper_bgcolor="#181818", plot_bgcolor="#181818",
                              font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data after filters.")
            
        st.markdown("""
        <div class='info-box'>
        <div class='title'>Insight</div>
        <div class='body'>
            - Most campaigns cluster around <b>$15–19k spend</b> generating <b>$10–30k revenue</b>. <br>
            - There’s at least one standout near <b>$22k spend</b> driving <b>~$75k revenue</b> (high-ROI outlier). <br>
            - Low-spend campaigns (<$6k) rarely exceed <b>$10k revenue</b>. <br>
            - Takeaway: Shift budget from low-yield clusters to the <b>top-left/upper</b> outliers (high revenue at moderate spend). Preserve reach only where it monetizes.
            
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>ROI by Format & Region</div>", unsafe_allow_html=True)
        st.markdown("<div class='caption'>Heatmap of average <b>ROI %</b> across ad formats and target regions.</div>", unsafe_allow_html=True)
        if not mrt_f.empty:
            grid = (mrt_f.groupby(["target_region","ad_format"], as_index=False)["roi_percentage"].mean())
            fig2 = px.density_heatmap(
                grid, x="ad_format", y="target_region", z="roi_percentage",
                color_continuous_scale="Reds"
            )
            fig2.update_layout(height=420, paper_bgcolor="#181818", plot_bgcolor="#181818",
                               font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data after filters.")
            
        st.markdown("""
        <div class='info-box'>
        <div class='title'>Insight</div>
        <div class='body'>
            - Average <b>ROI %</b> by <b>ad format</b> (video, carousel, banner) and <b>region</b> (US, EU, APAC). Darker cells = higher ROI. <br>
            - <b>EU + Video</b> is the strongest ROI pocket (darkest cell). <br>
            - <b>US + Banner</b> is the weakest (lightest cell). <br>
            - <b>Carousel</b> performs <b>mid-pack</b>, with <b>APAC</b> often stronger than <b>US</b>. <br>
            - Move dollars toward <b>high-ROI format×region combos</b> (especially <b>EU Video</b>).
            
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Daily trends (Spend, Revenue, Impr/Clicks, Efficiency)
    def daily_spend_from_campaigns(camps: pd.DataFrame) -> pd.DataFrame:
        rows=[]
        for _,r in camps.iterrows():
            if pd.isna(r.start_date) or pd.isna(r.end_date): 
                continue
            dts = pd.date_range(r.start_date, r.end_date, freq="D")
            if len(dts)==0: 
                continue
            per_day = float(r.budget or 0)/len(dts)
            rows += [[d.date(), per_day] for d in dts]
        if not rows:
            return pd.DataFrame(columns=["day","spend"])
        return pd.DataFrame(rows, columns=["day","spend"]).groupby("day", as_index=False).sum()

    spend_daily = daily_spend_from_campaigns(campaigns)
    daily = spend_daily.merge(impr_daily, on="day", how="outer") \
                       .merge(click_daily, on="day", how="outer") \
                       .merge(rev_daily, on="day", how="outer") \
                       .fillna(0.0)
    if not daily.empty:
        daily["cpc"] = np.where(daily["clicks"]>0, daily["spend"]/daily["clicks"], np.nan)
        daily["ctr"] = np.where(daily["impressions"]>0, daily["clicks"]/daily["impressions"], np.nan)
        daily = daily.sort_values("day")

        a,b = st.columns(2)
        with a:
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='h2'>Daily Spend vs Revenue</div>", unsafe_allow_html=True)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=daily["day"], y=daily["spend"], name="Spend", marker_color="#E50914"),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=daily["day"], y=daily["revenue"], name="Revenue", mode="lines+markers",
                                     line=dict(color="#1DB954")), secondary_y=True)
            fig.update_layout(height=360, paper_bgcolor="#181818", plot_bgcolor="#181818",
                              font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
            fig.update_yaxes(title_text="Spend", secondary_y=False)
            fig.update_yaxes(title_text="Revenue", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div class='info-box'>
            <div class='title'>Insight</div>
            <div class='body'>
                - Tracks <b>daily spend</b> (bars) against <b>daily revenue</b> (line) to reveal pacing and efficiency. <br>
                - Spend rises from <b>late June</b> to <b>mid-August</b>, peaking around <b>$23 K/day</b>. <br>
                - Revenue mirrors that curve closely, proving that investment directly translated into returns. <br>
                - After <b>August 20 to September</b>, both decline, suggesting <b>campaign wind-down</b> or <b>audience saturation</b>.
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with b:
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='h2'>Impressions → Clicks → CTR</div>", unsafe_allow_html=True)
            fig2 = go.Figure()
                # Bar trace — Impressions (left axis)
            fig2.add_trace(go.Bar(
                x=daily["day"],
                y=daily["impressions"],
                name="Impressions",
                marker_color="#444",
                opacity=0.8,
                yaxis="y1"
            ))

            # Line trace — CTR (right axis)
            fig2.add_trace(go.Scatter(
                x=daily["day"],
                y=daily["ctr"] * 100,  # convert CTR (0–1) to %
                name="CTR (%)",
                mode="lines",
                line=dict(color="#E50914", width=2),
                yaxis="y2"
            ))

            # Layout adjustments
            fig2.update_layout(
                height=360,
                paper_bgcolor="#181818",
                plot_bgcolor="#181818",
                font_color="#FFFFFF",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(x=0.02, y=0.98),
                xaxis=dict(title=None),
                yaxis=dict(
                    title="Impressions",
                    titlefont=dict(color="#AAAAAA"),
                    tickfont=dict(color="#AAAAAA")
                ),
                yaxis2=dict(
                    title="CTR (%)",
                    titlefont=dict(color="#E50914"),
                    tickfont=dict(color="#E50914"),
                    overlaying="y",
                    side="right",
                    showgrid=False
                )
            )

            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("""
            <div class='info-box'>
            <div class='title'>Insight</div>
            <div class='body'>
                - This chart shows how <b>ad visibility (impressions)</b> relates to <b>user engagement (CTR %)</b> over time. <br>
                - Impressions ramped up through <b>July–August</b>, peaking at around <b>12K/day</b>, while CTR stayed steady around <b>4–5%</b>, indicating consistent ad appeal even under higher reach. <br>
                - A gradual CTR dip through late August signals mild <b>creative fatigue</b> as exposure frequency increased. <br>
                - The <b>spikes in late September</b> suggest a brief resurgence, likely due to <b>new creatives or retargeting optimizations</b>. <br>
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h2'>Conversion Funnel</div>", unsafe_allow_html=True)
    st.markdown("<div class='caption'>From awareness to action — total activity during the period.</div>", unsafe_allow_html=True)
    if not mrt_f.empty:
        total_impr_f   = ssum(mrt_f["impressions_count"])
        total_clicks_f = ssum(mrt_f["clicks_count"])
        total_conv_f   = ssum(mrt_f["conversions_count"])
        figf = go.Figure(go.Funnel(
            y=["Impressions","Clicks","Conversions"],
            x=[total_impr_f, total_clicks_f, total_conv_f],
            marker={"color":["#3A3A3A","#E50914","#1DB954"]}
        ))
        figf.update_layout(height=320, paper_bgcolor="#181818", plot_bgcolor="#181818",
                           font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(figf, use_container_width=True)
    else:
        st.info("No data after filters.")
        
    st.markdown("""
<div class='info-box'>
<div class='title'>Insight</div>
<div class='body'>
    - Visualizes audience flow from <b>Awareness → Engagement → Action</b>. <br>
    - Out of <b>652 K impressions</b>, <b>30 K users clicked</b> → <b>CTR ≈ 4.6%</b>. <br>
    - Of those, <b>~1.1 K converted</b> → <b>Click-to-Conversion ≈ 3.8%</b>; overall <b>Impression-to-Conversion ≈ 0.18%</b>. <br>
    - The funnel confirms healthy post-click performance, with low drop-off between click and conversion.
</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# 2) CAMPAIGN INTELLIGENCE
# ======================================================
elif page.startswith("2"):
    st.markdown("<div class='h1'></div>", unsafe_allow_html=True)
    st.markdown("<div class='h1'>📈 Campaign Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<div class='caption'>Identify winners, scale efficiently, and spot waste.</div>", unsafe_allow_html=True)

    if mrt_f.empty:
        st.info("No data after filters.")
    else:
        # Top by ROI%
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>Top Campaigns by ROI %</div>", unsafe_allow_html=True)
        top = mrt_f.sort_values("roi_percentage", ascending=False).head(15)
        fig = px.bar(top, x="roi_percentage", y="advertiser", color="roi_percentage",
                     color_continuous_scale="Reds", orientation="h",
                     hover_data=["campaign_id","ctr","cvr","roas","total_revenue","budget"])
        fig.update_layout(height=450, paper_bgcolor="#181818", plot_bgcolor="#181818",
                          font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='caption'>Higher ROI means more revenue per dollar of spend. Use this to reallocate budget.</div>", unsafe_allow_html=True)
        st.markdown("""
<div class='info-box'>
<div class='title'>Insight</div>
<div class='body'>
    - Highlights the <b>top-performing advertisers</b> by <b>ROI %</b>, where ROI measures revenue per dollar of ad spend. <br>
    - <b>Stout Group</b> and <b>Delgado, Gonzales & Austin</b> dominate the chart with ROI values above <b>300%</b>, meaning each $1 spent brought in over $3 in returns. <br>
    - Mid-tier advertisers like <b>Hurley LLC</b> and <b>Fox, Morgan & Williams</b> maintain a steady 180–220% ROI, signaling efficient but scalable performance. <br>
    - Advertisers at the bottom (<b>Williams–Davis, Hendrix Inc</b>) show potential budget inefficiencies or under-optimized creatives. <br>
</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Spend vs Revenue — efficiency surface
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>Spend vs Revenue (bubble = Impressions)</div>", unsafe_allow_html=True)
        fig2 = px.scatter(mrt_f, x="budget", y="total_revenue", size="impressions_count",
                          color="ad_format", hover_name="advertiser",
                          labels={"budget":"Spend","total_revenue":"Revenue"},
                          color_discrete_sequence=["#E50914","#1DB954","#F59E0B"])
        fig2.update_layout(height=420, paper_bgcolor="#181818", plot_bgcolor="#181818",
                           font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
<div class='info-box'>
<div class='title'>Insight</div>
<div class='body'>
    - Each bubble represents a campaign, with size showing <b>impressions</b> and color showing <b>ad format</b> (video, carousel, banner). <br>
    - Most campaigns cluster between <b>$6K–$18K spend</b> and <b>$10K–$30K revenue</b>, revealing a performance plateau zone. <br>
    - Outliers near <b>$20K spend generating ~$75K revenue</b> mark <b>high-ROI opportunities</b>. <br>
    - <b>Video ads</b> generally yield higher revenue per spend than <b>banners</b> or <b>carousels</b>, proving richer creatives drive stronger conversion impact. <br>
    - Action: <b>shift budget</b> to top-performing formats or clusters that show disproportionately higher revenue for similar spend.
</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ML-guided targeting
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>Where should we scale next? (Predicted CTR)</div>", unsafe_allow_html=True)
        st.markdown("<div class='caption'>Average <b>Predicted CTR</b> from the model by Format → Region.</div>", unsafe_allow_html=True)
        if not pred.empty:
            grp = pred.groupby(["ad_format","target_region"], as_index=False)["predicted_prob"].mean()
            fig3 = px.treemap(grp, path=["ad_format","target_region"], values="predicted_prob",
                              color="predicted_prob", color_continuous_scale="Reds",
                              labels={"predicted_prob":"Predicted CTR"})
            fig3.update_layout(height=440, paper_bgcolor="#181818", plot_bgcolor="#181818",
                               font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("CTR predictions table is empty.")
            
        st.markdown("""
<div class='info-box'>
<div class='title'>Insight</div>
<div class='body'>
    - The <b>Video</b> format in <b>EU</b> and <b>US</b> regions shows the darkest shades, signaling the highest expected engagement. <br>
    - <b>Carousels</b> maintain mid-level CTR potential in <b>APAC</b> and <b>US</b>, making them cost-effective options for secondary scaling. <br>
    - <b>Banners</b> underperform globally, especially in <b>US</b> and <b>EU</b>, indicating creative fatigue or lower ad visibility. <br>
    - Recommendation: <b>scale EU and US Video inventory</b> first, followed by <b>carousel refreshes</b> in APAC for incremental growth.
</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Pareto (who pays the bills)
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>Revenue Concentration (Pareto)</div>", unsafe_allow_html=True)
        dfp = (mrt_f.groupby("advertiser", as_index=False)["total_revenue"].sum()
               .sort_values("total_revenue", ascending=False))
        if not dfp.empty:
            dfp["cum_pct"] = 100*dfp["total_revenue"].cumsum()/dfp["total_revenue"].sum()
            figp = make_subplots(specs=[[{"secondary_y": True}]])
            figp.add_trace(go.Bar(x=dfp["advertiser"], y=dfp["total_revenue"], name="Revenue", marker_color="#E50914"),
                           secondary_y=False)
            figp.add_trace(go.Scatter(x=dfp["advertiser"], y=dfp["cum_pct"], name="Cumulative %",
                                      line=dict(color="#1DB954")), secondary_y=True)
            figp.update_layout(height=380, paper_bgcolor="#181818", plot_bgcolor="#181818",
                               font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
            figp.update_yaxes(title_text="Revenue", secondary_y=False)
            figp.update_yaxes(title_text="Cumulative %", range=[0,100], secondary_y=True)
            st.plotly_chart(figp, use_container_width=True)
        else:
            st.info("No revenue rows.")
            
        st.markdown("""
<div class='info-box'>
<div class='title'>Insight</div>
<div class='body'>
    - The <b>first 5 advertisers</b> account for nearly <b>50% of total ad revenue</b>, led by <b>Stout Group</b> and <b>Rojas, Gardner & Wells</b>. <br>
    - The green cumulative line emphasizes <b>diminishing returns</b>, after the top 20 advertisers, each adds marginal revenue impact. <br>
    - Helps identify <b>key accounts to retain and prioritize</b> in renewal or upsell conversations. <br>
    - Strategic takeaway: focus retention and cross-sell efforts on the <b>top 20%</b> of accounts driving the majority of total revenue.
</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# 3) PREDICTIVE & AUDIENCE INSIGHTS
# ======================================================
else:
    st.markdown("<div class='h1'></div>", unsafe_allow_html=True)
    st.markdown("<div class='h1'>🧠 Predictive & Audience Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='caption'>Use ML to prioritize inventory and understand audience value.</div>", unsafe_allow_html=True)

    # CTR score distribution + feature importance
    l, r = st.columns([1.25, 1])
    with l:
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>CTR Score Distribution</div>", unsafe_allow_html=True)
        if not pred.empty:
            fig = px.histogram(pred, x="predicted_prob", nbins=40, color_discrete_sequence=["#E50914"])
            fig.update_layout(height=330, paper_bgcolor="#181818", plot_bgcolor="#181818",
                              font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='caption'>Right-skewed = more high-probability inventory; left-skewed = harder to convert.</div>",
                        unsafe_allow_html=True)
        else:
            st.info("No predictions available.")
            
        st.markdown("""
<div class='info-box'>
<div class='title'>Insight</div>
<div class='body'>
    - Shows the <b>predicted click-through probability</b> from the ML model for every impression. <br>
    - The histogram is slightly <b>right-skewed</b>, meaning most impressions have moderate-to-high predicted CTR scores (0.3–0.6). <br>
    - This suggests a <b>strong model bias toward mid-performing inventory</b> rather than overly confident predictions. <br>
    - Fewer low-probability (0–0.1) cases indicate the model has learned to <b>filter out poor-performing ad slots</b>. <br>
    - <b>Action:</b> Inventory with high predicted CTR (>0.6) can be prioritized in auctions or targeted for premium bids.
</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r:
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>Feature Importance (Top 10)</div>", unsafe_allow_html=True)
        if not feat.empty:
            fig = px.bar(feat.head(10), x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Reds")
            fig.update_layout(height=330, paper_bgcolor="#181818", plot_bgcolor="#181818",
                              font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='caption'>These are the strongest drivers of clicks. Use them to design experiments.</div>",
                        unsafe_allow_html=True)
        else:
            st.info("No feature importance data.")
        st.markdown("""
<div class='info-box'>
<div class='title'>Insight</div>
<div class='body'>
    - Ranks the <b>most influential features</b> driving predicted clicks in the CTR model. <br>
    - <b>Ad format</b> (0.43 importance) dominates, confirming that creative type (video, banner, carousel) has the highest impact on engagement. <br>
    - <b>User click rate</b> and <b>campaign CTR history</b> follow closely, showing that <b>past behavior strongly predicts future engagement</b>. <br>
    - <b>Budget</b> and <b>target region</b> carry moderate influence, meaning geographic and spend factors affect results but are less decisive. <br>
    - <b>Action:</b> Use these signals to design experiments — e.g., focus A/B tests on <b>ad format</b> and <b>user engagement history</b> first, since they drive most variance.
</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

#     # Calibration by decile
#     # st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.markdown("<div class='h2'>CTR Model Calibration (by Score Decile)</div>", unsafe_allow_html=True)
#     if not pred.empty:
#         q = pd.qcut(pred["predicted_prob"], 10, labels=False, duplicates="drop")
#         calib = pred.assign(decile=q).groupby("decile", as_index=False).agg(
#             predicted=("predicted_prob","mean"),
#             actual=("actual_clicked","mean"),
#             volume=("predicted_prob","count")
#         )
#         calib["decile"] = calib["decile"] + 1
#         figc = go.Figure()
#         figc.add_trace(go.Scatter(x=calib["decile"], y=calib["predicted"], mode="lines+markers",
#                                   name="Predicted CTR", line=dict(color="#E50914")))
#         figc.add_trace(go.Scatter(x=calib["decile"], y=calib["actual"], mode="lines+markers",
#                                   name="Actual CTR", line=dict(color="#1DB954")))
#         figc.update_layout(height=360, paper_bgcolor="#181818", plot_bgcolor="#181818",
#                            font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10),
#                            xaxis_title="Score Decile (1=low, 10=high)", yaxis_title="CTR")
#         st.plotly_chart(figc, use_container_width=True)
#         st.markdown("<div class='caption'>Lines should track each other. Divergence indicates miscalibration at certain score ranges.</div>",
#                     unsafe_allow_html=True)
#     else:
#         st.info("No predictions to calibrate.")
#     st.markdown("""
# <div class='info-box'>
# <div class='title'>Insight</div>
# <div class='body'>
#     - Validates how well the <b>model’s predicted CTR</b> aligns with the <b>actual observed CTR</b> across deciles (1 = lowest, 10 = highest). <br>
#     - Ideally, the <b>red (predicted)</b> and <b>green (actual)</b> lines should track closely — divergence signals <b>miscalibration</b>. <br>
#     - In this chart, the model consistently <b>overpredicts CTR</b>, especially in higher deciles — it’s optimistic for high-scoring impressions. <br>
#     - This pattern indicates <b>underfitting in the calibration layer</b> or the need for a <b>Platt scaling / isotonic regression</b> adjustment. <br>
#     - <b>Action:</b> Recalibrate using validation data to ensure predicted probabilities match real-world CTR outcomes.
# </div>
# </div>
# """, unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)

    # Audience Segments
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h2'>Audience Segments</div>", unsafe_allow_html=True)
    if not segments.empty:
        agg = (segments.groupby("user_cluster", as_index=False)
               .agg(users=("user_id","count"),
                    impressions=("impressions","sum"),
                    clicks=("clicks","sum"),
                    conversions=("conversions","sum"),
                    revenue=("total_revenue","sum"),
                    avg_ctr_score=("avg_ctr_score","mean")))
        a,b = st.columns(2)
        with a:
            figp = px.pie(agg, values="users", names="user_cluster", hole=0.45,
                          color_discrete_sequence=["#E50914","#1DB954","#F59E0B","#3B82F6","#A855F7"])
            figp.update_layout(height=340, paper_bgcolor="#181818", plot_bgcolor="#181818",
                               font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(figp, use_container_width=True)
            
        with b:
            figb = px.bar(agg.sort_values("revenue", ascending=False), x="user_cluster", y="revenue",
                          color="revenue", color_continuous_scale="Reds")
            figb.update_layout(height=340, paper_bgcolor="#181818", plot_bgcolor="#181818",
                               font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10),
                               xaxis_title="Cluster", yaxis_title="Revenue")
            st.plotly_chart(figb, use_container_width=True)

        st.markdown("""
<div class='info-box'>
<div class='title'>Insight</div>
<div class='body'>
    - Segments users into 4–5 behavioral clusters based on impressions, clicks, conversions, and revenue contribution. <br>
    - <b>Cluster 0 (Dormant)</b> forms nearly half of the base (47.7%) but contributes minimal revenue, a clear reactivation opportunity. <br>
    - <b>Cluster 3 (High Value)</b> drives the majority of total revenue (~$250M), with fewer users but higher monetization per head. <br>
    - <b>Cluster 2 (Earners)</b> maintain consistent CTR and solid conversion rates, the backbone audience for steady campaigns. <br>
    - <b>Action:</b> Allocate retention budgets to <b>Cluster 3</b>, and nurture <b>Cluster 2</b> via loyalty or frequency cap adjustments.
</div>
</div>
""", unsafe_allow_html=True)
        
        # Region × Device heatmap per cluster (top 3 clusters by revenue)
        top_clusters = agg.sort_values("revenue", ascending=False)["user_cluster"].head(3).tolist()
        grid = (segments[segments["user_cluster"].isin(top_clusters)]
                .groupby(["user_cluster","region","device"], as_index=False)["user_id"].count()
                .rename(columns={"user_id":"users"}))
        st.markdown("<div class='h2'>Where do high-value segments live?</div>", unsafe_allow_html=True)
        if not grid.empty:
            figh = px.density_heatmap(grid, x="device", y="region", z="users", facet_col="user_cluster",
                                      color_continuous_scale="Reds")
            figh.update_layout(height=380, paper_bgcolor="#181818", plot_bgcolor="#181818",
                               font_color="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(figh, use_container_width=True)
        else:
            st.info("Not enough data to map region × device by cluster.")
        st.markdown("""
<div class='info-box'>
<div class='title'>Insight</div>
<div class='body'>
    - Maps user clusters by <b>region × device</b> to reveal concentration of high-value audiences. <br>
    - <b>Mobile + US</b> emerges as the densest segment, the top-performing combination by user count and revenue impact. <br>
    - <b>EU + Mobile</b> and <b>APAC + Desktop</b> show mid-level presence but weaker monetization — secondary targets for optimization. <br>
    - Sparse Tablet activity indicates low conversion value, deprioritize tablet-specific creatives. <br>
    - <b>Action:</b> Scale premium campaigns for <b>US Mobile audiences</b>, refine creatives for <b>EU regions</b>, and reduce spend on low-engagement devices.
</div>
</div>
""", unsafe_allow_html=True)

        # Personas (copy as executive notes)
        
        st.markdown("<div class='h2'>Personas (Executive Notes)</div>", unsafe_allow_html=True)
        st.markdown("""
<div class='info-box'>
<div class='title'>Insight</div>
<div class='body'>
    - Converts quantitative clusters into <b>qualitative personas</b> for executive storytelling. <br>
    - <b>Dormant</b> (Cluster 0): Low impressions and clicks — ideal for reactivation with homepage takeovers and stronger CTAs. <br>
    - <b>Casual Browsers</b> (Cluster 1): Regular traffic, modest CTR — perfect for carousel or time-based retargeting. <br>
    - <b>Earners</b> (Cluster 2): High CTR, consistent conversions — double down with budget reinforcement and personalized offers. <br>
    - <b>High Value</b> (Cluster 3): Premium users driving most ROI — maintain frequency and protect experience quality. <br>
    - <b>New & Curious</b> (Cluster 4): Recent signups with volatile CTR — nurture with onboarding flows and retargeting campaigns.
</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("No user_segments table found / it is empty.")
    st.markdown("</div>", unsafe_allow_html=True)