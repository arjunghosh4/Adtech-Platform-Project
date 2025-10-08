# app.py — Netflix × Amazon DSP Executive Dashboard (schema-safe)
# Works with tables:
# campaigns, impressions, clicks, conversions, mrt_campaign_summary,
# ctr_predictions, ctr_feature_importance

import os, numpy as np, pandas as pd, streamlit as st
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------ UI Setup ------------------
st.set_page_config(page_title="Netflix × Amazon DSP – Executive Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.kpi-card{border:1px solid #E5E7EB;border-radius:14px;padding:16px 18px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04);height:100%}
.kpi-title{font-size:13px;font-weight:600;color:#6B7280;letter-spacing:.3px;text-transform:uppercase}
.kpi-value{font-size:28px;font-weight:700;color:#111827;margin-top:6px}
.kpi-sub{font-size:12px;color:#6B7280}
.section-card{border:1px solid #E5E7EB;border-radius:14px;padding:14px 16px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.caption{color:#4B5563;font-size:13px;margin-top:6px}
hr.sep{border:none;border-top:1px solid #E5E7EB;margin:12px 0 6px 0}
</style>
""", unsafe_allow_html=True)

def kpi(col, title, value, sub=None):
    with col:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>{title}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"<div class='kpi-sub'>{sub or ''}</div></div>", unsafe_allow_html=True
        )

# ------------------ DB ------------------
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5433")
DB_NAME = os.getenv("PGDATABASE", "ads_db")
DB_USER = os.getenv("PGUSER", "admin")
DB_PASS = os.getenv("PGPASSWORD", "admin")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True
)

@st.cache_data(ttl=300)
def load(sql): 
    with engine.begin() as conn: 
        return pd.read_sql(text(sql), conn)

# Core tables
mrt = load("""
    select campaign_id, advertiser, target_region, ad_format,
           impressions_count, clicks_count, conversions_count,
           total_revenue, budget, ctr, cvr, roi_percentage, cpc, cpa, cpm, rpc, rpcv, roas,
           active_days, revenue_per_day
    from mrt_campaign_summary
""")
pred = load("""
    select region, device, subscription_tier, ad_format, target_region,
           budget, hour_of_day, day_of_week, is_peak_hour,
           user_click_rate, campaign_ctr_history,
           actual_clicked, predicted_prob, predicted_clicked
    from ctr_predictions
""")
feat = load("select feature, importance from ctr_feature_importance order by importance desc")
campaigns = load("select campaign_id, advertiser, budget::float as budget, start_date, end_date from campaigns")
impr_daily = load("select date_trunc('day', timestamp)::date as day, count(*)::bigint as impressions from impressions group by 1 order by 1")
click_daily = load("select date_trunc('day', timestamp)::date as day, count(*)::bigint as clicks from clicks group by 1 order by 1")
rev_daily = load("select date_trunc('day', timestamp)::date as day, sum(coalesce(revenue,0))::float as revenue from conversions group by 1 order by 1")

# Segmentation tables (safe if missing)
try:
    user_segments = load("""
        select user_id, region, device, subscription_tier,
               impressions::float, clicks::float, conversions::float,
               total_revenue::float, coalesce(avg_ctr_score,0)::float as avg_ctr_score,
               days_since_signup::float, user_cluster::int
        from user_segments
    """)
except Exception:
    user_segments = pd.DataFrame()

try:
    cluster_summary = load("""
        select user_cluster::int,
               impressions::float, clicks::float, conversions::float,
               total_revenue::float, coalesce(avg_ctr_score,0)::float as avg_ctr_score,
               days_since_signup::float
        from user_cluster_summary
        order by user_cluster
    """)
except Exception:
    cluster_summary = pd.DataFrame()

# ------------------ Filters ------------------
st.sidebar.header("Filters")
region_f = st.sidebar.multiselect("Region", sorted(mrt["target_region"].dropna().unique().tolist()))
format_f = st.sidebar.multiselect("Ad Format", sorted(mrt["ad_format"].dropna().unique().tolist()))
adv_f    = st.sidebar.multiselect("Advertiser", sorted(mrt["advertiser"].dropna().unique().tolist()))

mask = pd.Series(True, index=mrt.index)
if region_f: mask &= mrt["target_region"].isin(region_f)
if format_f: mask &= mrt["ad_format"].isin(format_f)
if adv_f:    mask &= mrt["advertiser"].isin(adv_f)
mrt_f = mrt.loc[mask].copy()

def ssum(s): return float(pd.to_numeric(s, errors="coerce").fillna(0).sum())
def smean(s):
    x = pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
    return float(x.mean()) if len(x) else 0.0

total_impr = ssum(mrt_f["impressions_count"])
total_clicks = ssum(mrt_f["clicks_count"])
total_rev = ssum(mrt_f["total_revenue"])
total_cost = ssum(mrt_f["budget"])
avg_ctr = smean(mrt_f["ctr"])
avg_cvr = smean(mrt_f["cvr"])
avg_cpc = smean(mrt_f["cpc"])
avg_cpm = smean(mrt_f["cpm"])
avg_roi = smean(mrt_f["roi_percentage"])
avg_roas = smean(mrt_f["roas"])

page = st.sidebar.radio("Pages",
    ["1) Executive Summary","2) Top Campaigns","3) Performance Trend",
     "4) Predictive Insights","5) Ad-hoc Deep Dive","6) Audience Segmentation"], index=0)

# ------------------ Page 1 ------------------
if page.startswith("1"):
    st.title("Executive Summary")
    st.caption("Reach (Awareness), Engagement and Investment Efficiency for Netflix Ads via Amazon DSP.")

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"Total Impressions", f"{int(total_impr):,}", "Awareness")
    kpi(c2,"Total Clicks", f"{int(total_clicks):,}", "Engagement")
    kpi(c3,"Ad Revenue", f"${total_rev:,.0f}", "Attributed conversions")
    kpi(c4,"Ad Spend", f"${total_cost:,.0f}", "Campaign budgets")

    c5,c6,c7,c8 = st.columns(4)
    kpi(c5,"CTR", f"{(avg_ctr*100):.2f}%", "Ad appeal")
    kpi(c6,"CVR", f"{(avg_cvr*100):.2f}%", "Click → conversion")
    kpi(c7,"CPC", f"${avg_cpc:,.2f}", "Cost per click")
    kpi(c8,"CPM", f"${avg_cpm:,.2f}", "Cost per 1k impressions")

    st.markdown("<hr class='sep'/>", unsafe_allow_html=True)

    left,right = st.columns([1.2,1])
    with left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Who’s driving dollars?")
        st.markdown("<span class='caption'>Total **Ad Revenue** vs **Spend** by advertiser – bubble size = impressions.</span>", unsafe_allow_html=True)
        if not mrt_f.empty:
            df = (mrt_f.groupby("advertiser", as_index=False)
                        .agg(revenue=("total_revenue","sum"),
                             spend=("budget","sum"),
                             impressions=("impressions_count","sum")))
            fig = px.scatter(df, x="spend", y="revenue", size="impressions",
                             hover_name="advertiser", color="spend",
                             labels={"spend":"Ad Spend","revenue":"Ad Revenue"})
            fig.update_layout(height=420, margin=dict(l=20,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data after filters.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("ROI by Format")
        st.markdown("<span class='caption'>Distribution of **ROI %** across ad formats.</span>", unsafe_allow_html=True)
        if not mrt_f.empty:
            fig2 = px.box(mrt_f, x="ad_format", y="roi_percentage", points=False)
            fig2.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10),
                               yaxis_title="ROI %", xaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data after filters.")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Page 2 ------------------
elif page.startswith("2"):
    st.title("Top Campaigns")
    st.caption("Compare Awareness → Engagement → Efficiency. Use predicted CTR to guide scaling.")

    if mrt_f.empty:
        st.info("No data after filters.")
    else:
        df = mrt_f.copy().sort_values("total_revenue", ascending=False)
        for c in ["impressions_count","clicks_count","total_revenue","budget"]:
            m = df[c].max() or 1
            df[f"{c}_bar"] = (df[c]/m)*100

        def bar(v):
            return f"<div style='height:10px;background:#E5E7EB;border-radius:6px'><div style='height:10px;width:{v:.1f}%;background:#2563EB;border-radius:6px'></div></div>"

        show = df[["campaign_id","advertiser","target_region","ad_format",
                   "impressions_count","impressions_count_bar",
                   "clicks_count","clicks_count_bar",
                   "ctr","cpc","roas","roi_percentage",
                   "total_revenue","total_revenue_bar",
                   "budget","budget_bar"]]

        def render(pdf):
            rows=[]
            head="""<table style='width:100%;border-collapse:collapse'>
            <thead><tr style='text-align:left;border-bottom:1px solid #E5E7EB'>
            <th>Campaign</th><th>Advertiser</th><th>Region</th><th>Format</th>
            <th style='width:14%'>Impr.</th><th style='width:16%'></th>
            <th style='width:12%'>Clicks</th><th style='width:16%'></th>
            <th>CTR</th><th>CPC</th><th>ROAS</th><th>ROI %</th>
            <th style='width:14%'>Revenue</th><th style='width:16%'></th>
            <th style='width:14%'>Budget</th><th style='width:16%'></th></tr></thead><tbody>"""
            for _,r in pdf.iterrows():
                rows.append(f"<tr style='border-bottom:1px solid #F3F4F6'>"
                            f"<td>{r.campaign_id}</td><td>{r.advertiser}</td><td>{r.target_region}</td><td>{r.ad_format}</td>"
                            f"<td>{int(r.impressions_count):,}</td><td>{bar(r.impressions_count_bar)}</td>"
                            f"<td>{int(r.clicks_count):,}</td><td>{bar(r.clicks_count_bar)}</td>"
                            f"<td>{r.ctr:.3f}</td><td>${r.cpc:,.2f}</td><td>{r.roas:.2f}</td><td>{r.roi_percentage:.2f}</td>"
                            f"<td>${r.total_revenue:,.0f}</td><td>{bar(r.total_revenue_bar)}</td>"
                            f"<td>${r.budget:,.0f}</td><td>{bar(r.budget_bar)}</td></tr>")
            return head + "".join(rows) + "</tbody></table>"

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Campaign Performance")
        st.markdown(render(show.head(30)), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Where can we scale efficiently?")
        st.markdown("<span class='caption'>Average **Predicted CTR** by **Ad Format** & **Region** (from model scores).</span>", unsafe_allow_html=True)
        if not pred.empty:
            grp = pred.groupby(["ad_format","target_region"], as_index=False)["predicted_prob"].mean()
            fig = px.treemap(grp, path=["ad_format","target_region"], values="predicted_prob",
                             color="predicted_prob", color_continuous_scale="Blues",
                             labels={"predicted_prob":"Predicted CTR"})
            fig.update_layout(height=440, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("CTR predictions table is empty.")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Page 3 (fixed) ------------------
elif page.startswith("3"):
    st.title("Performance Trend")
    st.caption("Daily Spend (from campaigns), Reach/Clicks (from logs) and Efficiency over time.")

    # Build daily spend from campaigns (allocate evenly over active days)
    def daily_spend_from_campaigns(camps: pd.DataFrame) -> pd.DataFrame:
        rows=[]
        for _,r in camps.iterrows():
            if pd.isna(r.start_date) or pd.isna(r.end_date): 
                continue
            dts = pd.date_range(r.start_date, r.end_date, freq="D")
            if len(dts)==0: 
                continue
            per_day = float(r.budget or 0)/len(dts)
            for d in dts:
                rows.append([d.date(), per_day])
        if not rows:
            return pd.DataFrame(columns=["day","spend"])
        return pd.DataFrame(rows, columns=["day","spend"]).groupby("day", as_index=False).sum()

    spend_daily = daily_spend_from_campaigns(campaigns)

    # Merge daily facts
    daily = spend_daily.merge(impr_daily, on="day", how="outer") \
                       .merge(click_daily, on="day", how="outer") \
                       .merge(rev_daily, on="day", how="outer") \
                       .fillna(0)
    if daily.empty:
        st.info("No sufficient daily data to display.")
    else:
        daily["cpc"] = np.where(daily["clicks"]>0, daily["spend"]/daily["clicks"], np.nan)
        daily["ctr"] = np.where(daily["impressions"]>0, daily["clicks"]/daily["impressions"], np.nan)
        daily = daily.sort_values("day")

        a,b = st.columns(2)
        with a:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Ad Spend (Daily)")
            st.plotly_chart(px.area(daily, x="day", y="spend").update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                           yaxis_title="Spend"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with b:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Revenue (Daily)")
            st.plotly_chart(px.area(daily, x="day", y="revenue").update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                           yaxis_title="Revenue"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c,d = st.columns(2)
        with c:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Impressions & Clicks")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=daily["day"], y=daily["impressions"], name="Impressions", opacity=0.5))
            fig.add_trace(go.Bar(x=daily["day"], y=daily["clicks"], name="Clicks", opacity=0.9))
            fig.update_layout(barmode="overlay", height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with d:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Efficiency Over Time")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=daily["day"], y=daily["cpc"], mode="lines", name="CPC"))
            fig2.add_trace(go.Scatter(x=daily["day"], y=daily["ctr"], mode="lines", name="CTR", yaxis="y2"))
            fig2.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                               yaxis=dict(title="CPC"),
                               yaxis2=dict(overlaying="y", side="right", title="CTR"))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Page 4 ------------------
elif page.startswith("4"):
    st.title("Predictive Insights (CTR Model)")
    st.caption("Use model scores to prioritize inventory.")

    left,right = st.columns([1.2,1])
    with left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("CTR Score Distribution")
        if not pred.empty:
            st.plotly_chart(px.histogram(pred, x="predicted_prob", nbins=40)
                            .update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10)),
                            use_container_width=True)
        else:
            st.info("No predictions available.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Feature Importance")
        if not feat.empty:
            st.plotly_chart(px.bar(feat.head(10), x="importance", y="feature", orientation="h")
                            .update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10)),
                            use_container_width=True)
        else:
            st.info("No feature importance table.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Calibration by Score Decile")
    if not pred.empty:
        q = pd.qcut(pred["predicted_prob"], 10, labels=False, duplicates="drop")
        dfc = pred.assign(decile=q)
        calib = dfc.groupby("decile", as_index=False).agg(
            predicted=("predicted_prob","mean"),
            actual=("actual_clicked","mean"),
            volume=("predicted_prob","count")
        )
        calib["decile"] = calib["decile"] + 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=calib["decile"], y=calib["predicted"], mode="lines+markers", name="Predicted CTR"))
        fig.add_trace(go.Scatter(x=calib["decile"], y=calib["actual"], mode="lines+markers", name="Actual CTR"))
        fig.update_layout(height=380, xaxis_title="Score Decile (1=low, 10=high)", yaxis_title="CTR",
                          margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions to calibrate.")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Page 6 ------------------
elif page.startswith("6"):
    st.title("Audience Segmentation")
    st.caption("Clusters of Netflix viewers by engagement & value, built from impression/click/conversion logs.")

    if user_segments.empty or cluster_summary.empty:
        st.info("Segmentation tables not found or empty. Make sure `user_segments` and `user_cluster_summary` exist.")
    else:
        # Pretty labels for clusters (tweak if you change K)
        CLUSTER_LABELS = {
            0: "Dormant",
            1: "Casual Viewers",
            2: "High Value",
            3: "Active Earners"
        }
        # attach labels
        cs = cluster_summary.copy()
        cs["cluster_name"] = cs["user_cluster"].map(CLUSTER_LABELS).fillna(cs["user_cluster"].astype(str))

        # KPI row
        total_users = len(user_segments)
        users_by_cluster = user_segments.groupby("user_cluster")["user_id"].nunique().reindex(cs["user_cluster"]).fillna(0).astype(int)
        total_revenue_seg = float(cs["total_revenue"].fillna(0).sum())
        avg_days_since_signup = float(user_segments["days_since_signup"].mean())

        c1,c2,c3,c4 = st.columns(4)
        kpi(c1, "Total Users", f"{total_users:,}", "in segmentation")
        kpi(c2, "Clusters", f"{cs['user_cluster'].nunique():,}", "K-means segments")
        kpi(c3, "Ad Revenue (segmented)", f"${total_revenue_seg:,.0f}", "sum of conversion revenue")
        kpi(c4, "Avg Days Since Signup", f"{avg_days_since_signup:,.0f}", "user recency proxy")

        st.markdown("<hr class='sep'/>", unsafe_allow_html=True)

        # Row: Share of users + Revenue by cluster
        a,b = st.columns([1.2,1])
        with a:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("User Share by Segment")
            st.markdown("<span class='caption'>How your audience breaks down by behavioral segments.</span>", unsafe_allow_html=True)
            pie_df = pd.DataFrame({
                "cluster": cs["cluster_name"].values,
                "users": users_by_cluster.values
            })
            fig = px.pie(pie_df, names="cluster", values="users", hole=0.35)
            fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with b:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Revenue & Conversions by Segment")
            st.markdown("<span class='caption'>Where dollars are coming from today—helps prioritize audiences in Amazon DSP.</span>", unsafe_allow_html=True)
            bar_df = cs[["cluster_name","total_revenue","conversions"]].copy()
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=bar_df["cluster_name"], y=bar_df["total_revenue"], name="Revenue"), secondary_y=False)
            fig2.add_trace(go.Scatter(x=bar_df["cluster_name"], y=bar_df["conversions"], mode="lines+markers",
                                      name="Conversions"), secondary_y=True)
            fig2.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
            fig2.update_yaxes(title_text="Revenue", secondary_y=False)
            fig2.update_yaxes(title_text="Conversions", secondary_y=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Row: Engagement depth + recency
        from plotly.subplots import make_subplots
        c,d = st.columns(2)
        with c:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Engagement Depth by Segment")
            st.markdown("<span class='caption'>Average impressions → clicks per user, by segment.</span>", unsafe_allow_html=True)
            eng = cs[["cluster_name","impressions","clicks"]].copy().melt("cluster_name", var_name="metric", value_name="value")
            st.plotly_chart(px.bar(eng, x="cluster_name", y="value", color="metric", barmode="group")
                            .update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10),
                                           xaxis_title="", yaxis_title="Avg per user"),
                            use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with d:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Recency by Segment")
            st.markdown("<span class='caption'>Lower days since signup = newer cohorts. Useful for targeting experiments.</span>", unsafe_allow_html=True)
            st.plotly_chart(px.bar(cs, x="cluster_name", y="days_since_signup", color="cluster_name",
                                   labels={"days_since_signup":"Avg days since signup", "cluster_name":""})
                            .update_layout(showlegend=False, height=340, margin=dict(l=10,r=10,t=10,b=10)),
                            use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


        # Profiles (plain-English cards)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Segment Profiles (what to do next)")
        st.markdown("""
        - **Dormant** – low activity, older signups. *Action:* low-cost retargeting, lightweight creatives.  
        - **Casual Viewers** – moderate impressions & some clicks, no $$ yet. *Action:* upsell tests, free-to-paid offers.  
        - **Active Earners** – consistent engagement & some revenue. *Action:* retention + personalized ad rotations.  
        - **High Value** – strong conversions and revenue. *Action:* premium inventory, lookalike expansion in Amazon DSP.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        # Export
        st.download_button("⬇️ Download user_segments CSV",
                           user_segments.to_csv(index=False).encode("utf-8"),
                           file_name="user_segments.csv", mime="text/csv")
        
# ------------------ Page 5 ------------------
else:
    st.title("Ad-hoc Deep Dive")
    st.caption("Explore and export filtered data.")

    if mrt_f.empty:
        st.info("No data after filters.")
    else:
        a,b = st.columns(2)
        with a:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Revenue by Advertiser (Pareto)")
            df = (mrt_f.groupby("advertiser", as_index=False)["total_revenue"].sum()
                        .sort_values("total_revenue", ascending=False))
            df["cum_pct"] = 100*df["total_revenue"].cumsum()/df["total_revenue"].sum()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["advertiser"], y=df["total_revenue"], name="Revenue"))
            fig.add_trace(go.Scatter(x=df["advertiser"], y=df["cum_pct"], name="Cumulative %", yaxis="y2"))
            fig.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10),
                              yaxis=dict(title="Revenue"),
                              yaxis2=dict(overlaying="y", side="right", range=[0,100], title="Cumulative %"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with b:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("ROI by Region & Format")
            grid = (mrt_f.groupby(["target_region","ad_format"], as_index=False)["roi_percentage"].mean())
            st.plotly_chart(px.density_heatmap(grid, x="ad_format", y="target_region", z="roi_percentage",
                                               color_continuous_scale="Blues")
                            .update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10)),
                            use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Raw Table (Filtered)")
        st.dataframe(mrt_f.sort_values("total_revenue", ascending=False), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    