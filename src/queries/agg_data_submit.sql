create table agg_data_submit as
    select
        user_id, date, part_of_day,
        avg(region_name_encoded) as region_name,
        avg(city_name_encoded) as city_name,
        avg(cpe_manufacturer_name_encoded) as cpe_manufacturer_name,
        avg(cpe_model_name_encoded) as cpe_model_name,
        avg(url_host_encoded) as url_host,
        avg(cpe_type_cd_encoded) as cpe_type_cd,
        avg(cpe_model_os_type_encoded) as cpe_model_os_type,
        avg(price) as price,
        avg(request_cnt) as request_cnt
    from
        data_prepared_submit
    group by
        user_id, date, part_of_day
