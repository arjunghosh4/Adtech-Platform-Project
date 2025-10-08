{{ config(materialized='view') }}

with
impressions as (
    select campaign_id, count(*) as impressions_count
    from {{ ref('stg_impressions') }}
    group by campaign_id
),
clicks as (
    select campaign_id, count(*) as clicks_count
    from {{ ref('stg_clicks') }}
    group by campaign_id
),
conversions as (
    select campaign_id,
           count(*) as conversions_count,
           coalesce(sum(revenue), 0) as total_revenue
    from {{ ref('stg_conversions') }}
    group by campaign_id
)

select
    c.campaign_id,
    i.impressions_count,
    cks.clicks_count,
    conv.conversions_count,
    conv.total_revenue
from {{ ref('stg_campaigns') }} c
left join impressions i on c.campaign_id = i.campaign_id
left join clicks cks on c.campaign_id = cks.campaign_id
left join conversions conv on c.campaign_id = conv.campaign_id