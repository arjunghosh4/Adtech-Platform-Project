{{ config(materialized='view') }}

WITH engagement AS (
    SELECT 
        u.user_id,
        COUNT(DISTINCT i.impression_id) AS impressions,
        COUNT(DISTINCT c.click_id) AS clicks,
        COUNT(DISTINCT v.conversion_id) AS conversions,
        COALESCE(SUM(v.revenue), 0) AS total_revenue
    FROM {{ ref('stg_users') }} u
    LEFT JOIN {{ ref('stg_impressions') }} i ON u.user_id = i.user_id
    LEFT JOIN {{ ref('stg_clicks') }} c ON u.user_id = c.user_id
    LEFT JOIN {{ ref('stg_conversions') }} v ON u.user_id = v.user_id
    GROUP BY u.user_id
),

ctr_behavior AS (
    SELECT 
        i.user_id,
        AVG(p.predicted_prob) AS avg_ctr_score
    FROM public.ctr_predictions p
    JOIN {{ ref('stg_impressions') }} i
      ON p.ad_format = i.ad_format
     AND p.device = i.device
     AND p.target_region = i.ad_slot  -- adjust if needed
    GROUP BY i.user_id
)

SELECT
    e.user_id,
    e.impressions,
    e.clicks,
    e.conversions,
    e.total_revenue,
    COALESCE(c.avg_ctr_score, 0) AS avg_ctr_score,
    u.region,
    u.device,
    u.subscription_tier,
    (CURRENT_DATE - u.signup_date) AS days_since_signup
FROM engagement e
JOIN {{ ref('stg_users') }} u ON e.user_id = u.user_id
LEFT JOIN ctr_behavior c ON e.user_id = c.user_id