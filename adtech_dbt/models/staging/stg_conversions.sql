{{ config(materialized='view') }}

select
    cast(conversion_id as bigint) as conversion_id,
    cast(user_id as integer) as user_id,
    cast(campaign_id as integer) as campaign_id,
    cast(timestamp as timestamp) as timestamp,
    product_category,
    cast(revenue as numeric) as revenue
from public.conversions
where user_id is not null and campaign_id is not null