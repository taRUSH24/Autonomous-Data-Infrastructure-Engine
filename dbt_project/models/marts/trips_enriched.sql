{{ config(materialized='table') }}

-- The join-fault target: pickup zone is resolved by joining stg_trips to
-- stg_zones on pickup_location_id. A corrupted join key (e.g. a fault that
-- overwrites pickup_location_id with an out-of-range value) silently drops
-- or fans out rows here without touching stg_trips row counts.

select
    t.vendor_id,
    t.pickup_at,
    t.dropoff_at,
    t.pickup_date,
    t.passenger_count,
    t.trip_distance_miles,
    t.pickup_location_id,
    t.dropoff_location_id,
    z.borough      as pickup_borough,
    z.zone         as pickup_zone,
    t.fare_amount,
    t.tip_amount,
    t.total_amount
from {{ ref('stg_trips') }} t
left join {{ ref('stg_zones') }} z
    on t.pickup_location_id = z.location_id
