{{
  config(materialized='view')
}}

-- Casts and lightly cleans the raw TLC feed. This is the model that an
-- "incorrect transformation" fault targets: a bug here changes a derived
-- column without touching the source, so plane-1 checks on fct_trips_daily
-- fire while raw.trips looks perfectly healthy.

with source as (
    select * from {{ source('raw', 'trips') }}
)

select
    cast(vendorid as integer)                      as vendor_id,
    cast(tpep_pickup_datetime as timestamp)         as pickup_at,
    cast(tpep_dropoff_datetime as timestamp)        as dropoff_at,
    cast(passenger_count as double)                 as passenger_count,
    cast(trip_distance as double)                   as trip_distance_miles,
    cast(ratecodeid as integer)                     as rate_code_id,
    store_and_fwd_flag,
    cast(pulocationid as integer)                   as pickup_location_id,
    cast(dolocationid as integer)                   as dropoff_location_id,
    cast(payment_type as integer)                   as payment_type,
    cast(fare_amount as double)                     as fare_amount,
    cast(extra as double)                           as extra_amount,
    cast(mta_tax as double)                         as mta_tax_amount,
    cast(tip_amount as double)                      as tip_amount,
    cast(tolls_amount as double)                    as tolls_amount,
    cast(improvement_surcharge as double)           as improvement_surcharge,
    cast(total_amount as double)                    as total_amount,
    cast(congestion_surcharge as double)            as congestion_surcharge,
    date_trunc('day', cast(tpep_pickup_datetime as timestamp)) as pickup_date
from source
where tpep_pickup_datetime is not null
