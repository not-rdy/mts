-- SQLite
create table data_prepared_submit as
    select
        region_name_encoded,
        city_name_encoded,
        cpe_manufacturer_name_encoded,
        cpe_model_name_encoded,
        url_host_encoded,
        cpe_type_cd_encoded,
        cpe_model_os_type_encoded,
        price,
        date,
        part_of_day,
        request_cnt,
        user_id
    from
        (
            select
                region_name,
                city_name,
                cpe_manufacturer_name,
                cpe_model_name,
                url_host,
                cpe_type_cd,
                cpe_model_os_type,
                price,
                date,
                part_of_day,
                request_cnt,
                user_id
            from
                data
            where
                age = 'NULL'
        ) as t1
        left join
        (
            select
                region_name,
                avg(age) as region_name_encoded
            from
                data
            where
                age != 'NULL'
            group by
                region_name
        ) as t2
        on t1.region_name = t2.region_name
        left join
        (
            select
                city_name,
                avg(age) as city_name_encoded
            from
                data
            where
                age != 'NULL'
            group by
                city_name
        ) as t3
        on t1.city_name = t3.city_name
        left join
        (
            select
                cpe_manufacturer_name,
                avg(age) as cpe_manufacturer_name_encoded
            from
                data
            where
                age != 'NULL'
            group by
                cpe_manufacturer_name
        ) as t4
        on t1.cpe_manufacturer_name = t4.cpe_manufacturer_name
        left join
        (
            select
                cpe_model_name,
                avg(age) as cpe_model_name_encoded
            from
                data
            where
                age != 'NULL'
            group by
                cpe_model_name
        ) as t5
        on t1.cpe_model_name = t5.cpe_model_name
        left join
        (
            select
                url_host,
                avg(age) as url_host_encoded
            from
                data
            where
                age != 'NULL'
            group by
                url_host
        ) as t6
        on t1.url_host = t6.url_host
        left join
        (
            select
                cpe_type_cd,
                avg(age) as cpe_type_cd_encoded
            from
                data
            where
                age != 'NULL'
            group by
                cpe_type_cd
        )  as t7
        on t1.cpe_type_cd = t7.cpe_type_cd
        left join
        (
        select
            cpe_model_os_type,
            avg(age) as cpe_model_os_type_encoded
        from
            data
        where
            age != 'NULL'
        group by
            cpe_model_os_type
        ) as t8
        on t1.cpe_model_os_type = t8.cpe_model_os_type

