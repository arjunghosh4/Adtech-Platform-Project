{{ config(materialized='view') }}

select
    cast(campaign_id as integer) as campaign_id,
    advertiser,
    cast(budget as numeric) as budget,
    initcap(target_region) as target_region,
    lower(ad_format) as ad_format,
    cast(start_date as date) as start_date,
    cast(end_date as date) as end_date
from public.campaigns
where campaign_id is not null