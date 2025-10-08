{{ config(materialized='view') }}

select
    cast(user_id as integer) as user_id,
    initcap(region) as region,
    initcap(device) as device,
    initcap(subscription_tier) as subscription_tier,
    cast(signup_date as date) as signup_date
from public.users
where user_id is not null