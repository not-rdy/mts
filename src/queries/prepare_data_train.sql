create table train_prepared as
    select
        region_name_encoded as region_name,
        city_name_encoded as city_name,
        cpe_manufacturer_name_encoded as cpe_manufacturer_name,
        cpe_model_name_encoded as cpe_model_name,
        url_host_encoded as url_host,
        cpe_type_cd_encoded as cpe_type_cd,
        cpe_model_os_type_encoded as cpe_model_os_type,
        price,
        date,
        part_of_day,
        request_cnt,
        user_id
    from
        (
        select
            *
        from
            data_train
        ) as t1
        left join
        (
        select
            distinct
                region_name,
                avg(age)
            over
                (partition by region_name) as region_name_encoded
        from
            data_train
        ) as t2
        on t1.region_name = t2.region_name
        left join
        (
        select
            distinct
                city_name,
                avg(age)
            over
                (partition by city_name) as city_name_encoded
        from
            data_train
        ) as t3
        on t1.city_name = t3.city_name
        left join
        (
        select
            distinct
                cpe_manufacturer_name,
                avg(age)
            over
                (partition by cpe_manufacturer_name) as cpe_manufacturer_name_encoded
        from
            data_train
        ) as t4
        on t1.cpe_manufacturer_name = t4.cpe_manufacturer_name
        left join
        (
        select
            distinct
                cpe_model_name,
                avg(age)
            over
                (partition by cpe_model_name) as cpe_model_name_encoded
        from
            data_train
        ) as t5
        on t1.cpe_model_name = t5.cpe_model_name
        left join
        (
        select
            distinct
                url_host,
                avg(age)
            over
                (partition by url_host) as url_host_encoded
        from
            data_train
        ) as t6
        on t1.url_host = t6.url_host
        left join
        (
        select
            distinct
                cpe_type_cd,
                avg(age)
            over
                (partition by cpe_type_cd) as cpe_type_cd_encoded
        from
            data_train
        ) as t7
        on t1.cpe_type_cd = t7.cpe_type_cd
        left join
        (
        select
            distinct
                cpe_model_os_type,
                avg(age)
            over
                (partition by cpe_model_os_type) as cpe_model_os_type_encoded
        from
            data_train
        ) as t8
        on t1.cpe_model_os_type = t8.cpe_model_os_type
