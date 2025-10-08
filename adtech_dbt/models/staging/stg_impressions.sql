{{ config(materialized='view') }}

select
    cast(impression_id as bigint) as impression_id,
    cast(user_id as integer) as user_id,
    cast(campaign_id as integer) as campaign_id,
    cast(timestamp as timestamp) as timestamp,
    lower(device) as device,
    lower(ad_slot) as ad_slot,
    lower(ad_format) as ad_format
from public.impressions
where user_id is not null and campaign_id is not null