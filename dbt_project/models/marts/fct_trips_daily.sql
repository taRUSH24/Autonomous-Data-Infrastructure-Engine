{{ config(materialized='table') }}

-- The asset every detector in plane 1 watches. Day-grain so that a single
-- bad day (a missing-partition or duplicate-records fault) is visible as one
-- clearly anomalous row rather than diluted into a monthly average.

select
    pickup_date,
    count(*)                                   as trip_count,
    count(distinct pickup_borough)              as distinct_boroughs,
    sum(case when pickup_borough is null then 1 else 0 end) as unmatched_zone_count,
    avg(fare_amount)                            as avg_fare_amount,
    avg(trip_distance_miles)                    as avg_trip_distance_miles,
    avg(total_amount)                           as avg_total_amount,
    sum(total_amount)                           as total_revenue,
    avg(tip_amount)                             as avg_tip_amount
from {{ ref('trips_enriched') }}
group by 1
order by 1
