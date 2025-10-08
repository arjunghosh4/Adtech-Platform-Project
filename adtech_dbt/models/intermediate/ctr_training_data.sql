{{ config(materialized='table') }}

WITH impression_base AS (
    SELECT 
        i.impression_id,
        i.user_id,
        i.campaign_id,
        u.region,
        u.device,
        u.subscription_tier,
        c.ad_format,
        c.target_region,
        c.budget,
        EXTRACT(HOUR FROM i.timestamp) AS hour_of_day,
        EXTRACT(DOW FROM i.timestamp) AS day_of_week,
        CASE WHEN EXTRACT(HOUR FROM i.timestamp) BETWEEN 18 AND 22 THEN 1 ELSE 0 END AS is_peak_hour,
        i.timestamp AS impression_time
    FROM {{ ref('stg_impressions') }} i
    JOIN {{ ref('stg_users') }} u ON i.user_id = u.user_id
    JOIN {{ ref('stg_campaigns') }} c ON i.campaign_id = c.campaign_id
),

click_enriched AS (
    SELECT 
        c.user_id,
        c.campaign_id,
        MIN(c.timestamp) AS first_click_time
    FROM {{ ref('stg_clicks') }} c
    GROUP BY c.user_id, c.campaign_id
),

-- 🧮 user-level historical engagement (clicks per campaign)
user_click_rate AS (
    SELECT 
        user_id,
        COUNT(DISTINCT campaign_id)::float / NULLIF(COUNT(DISTINCT impression_id),0) AS user_click_rate
    FROM {{ ref('stg_clicks') }} c
    JOIN {{ ref('stg_impressions') }} i USING (user_id, campaign_id)
    GROUP BY user_id
),

-- 🧮 campaign-level historical CTR
campaign_ctr_history AS (
    SELECT 
        campaign_id,
        COUNT(DISTINCT c.click_id)::float / NULLIF(COUNT(DISTINCT i.impression_id),0) AS campaign_ctr_history
    FROM {{ ref('stg_impressions') }} i
    LEFT JOIN {{ ref('stg_clicks') }} c USING (user_id, campaign_id)
    GROUP BY campaign_id
)

SELECT 
    ib.impression_id,
    ib.user_id,
    ib.campaign_id,
    ib.region,
    ib.device,
    ib.subscription_tier,
    ib.ad_format,
    ib.target_region,
    ib.budget,
    ib.hour_of_day,
    ib.day_of_week,
    ib.is_peak_hour,
    COALESCE(ucr.user_click_rate, 0) AS user_click_rate,
    COALESCE(cch.campaign_ctr_history, 0) AS campaign_ctr_history,
    CASE 
        WHEN ce.first_click_time IS NOT NULL 
             AND ce.first_click_time >= ib.impression_time 
        THEN 1 
        ELSE 0 
    END AS clicked
FROM impression_base ib
LEFT JOIN click_enriched ce
  ON ib.user_id = ce.user_id
  AND ib.campaign_id = ce.campaign_id
LEFT JOIN user_click_rate ucr
  ON ib.user_id = ucr.user_id
LEFT JOIN campaign_ctr_history cch
  ON ib.campaign_id = cch.campaign_id
WHERE ib.region IS NOT NULL
  AND ib.device IS NOT NULL
  AND ib.ad_format IS NOT NULL
  AND ib.target_region IS NOT NULL