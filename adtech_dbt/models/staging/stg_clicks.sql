{{ config(materialized='view') }}

select
    cast(click_id as bigint) as click_id,
    cast(user_id as integer) as user_id,
    cast(campaign_id as integer) as campaign_id,
    cast(timestamp as timestamp) as timestamp,
    cast(click_position as integer) as click_position
from public.clicks
where user_id is not null and campaign_id is not null