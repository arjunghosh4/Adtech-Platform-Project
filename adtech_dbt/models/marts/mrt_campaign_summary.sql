{{ config(materialized='view') }}

with performance as (
    select *
    from {{ ref('int_campaign_performance') }}
),
campaigns as (
    select campaign_id, advertiser, budget, target_region, ad_format,
           start_date, end_date
    from {{ ref('stg_campaigns') }}
)

select
    c.campaign_id,
    c.advertiser,
    c.target_region,
    c.ad_format,
    p.impressions_count,
    p.clicks_count,
    p.conversions_count,
    p.total_revenue,
    c.budget,
    round((p.clicks_count::numeric / nullif(p.impressions_count, 0)), 4) as ctr,
    round((p.conversions_count::numeric / nullif(p.clicks_count, 0)), 4) as cvr,
    round(((p.total_revenue - c.budget) / nullif(c.budget, 0)) * 100, 2) as roi_percentage,
    round((c.budget / nullif(p.clicks_count, 0)), 2) as cpc,
    round((c.budget / nullif(p.conversions_count, 0)), 2) as cpa,
    round(((c.budget / nullif(p.impressions_count, 0)) * 1000), 2) as cpm,
    round((p.total_revenue / nullif(p.clicks_count, 0)), 2) as rpc,
    round((p.total_revenue / nullif(p.conversions_count, 0)), 2) as rpcv,
    round((p.total_revenue / nullif(c.budget, 0)), 2) as roas,
    (c.end_date - c.start_date) as active_days,
    round((p.total_revenue / nullif((c.end_date - c.start_date), 0)), 2) as revenue_per_day
from performance p
join campaigns c on p.campaign_id = c.campaign_id