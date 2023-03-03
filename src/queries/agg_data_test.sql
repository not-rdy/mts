create table test_agg as
    select
        user_id, date, part_of_day,
        avg(region_name) as region_name,
        avg(city_name) as city_name,
        avg(cpe_manufacturer_name) as cpe_manufacturer_name,
        avg(cpe_model_name) as cpe_model_name,
        avg(url_host) as url_host,
        avg(cpe_type_cd) as cpe_type_cd,
        avg(cpe_model_os_type) as cpe_model_os_type,
        avg(price) as price,
        avg(request_cnt) as request_cnt
    from
        test_prepared
    group by
        user_id, date, part_of_day
