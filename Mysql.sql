select column_names as names from (select fuction() as name from table_name group by column_name4 )
  where column_name like '%*%' order by column_name5 desc
select date_add(week(now()),interval n week)#日期加减
select datediff(date1,date2)/timediff(time1,time2)#时间间隔
select substr(salary,locate('-',salary)+1,length(salary)-locate('-',salary)-1)#数据清洗获取中间字段

union 连接查询结果
连接类型	说明
INNER JOIN	（默认连接方式）只有当两个表都存在满足条件的记录时才会返回行。
LEFT JOIN	返回左表中的所有行，即使右表中没有满足条件的行也是如此。
RIGHT JOIN	返回右表中的所有行，即使左表中没有满足条件的行也是如此。
FULL JOIN	只要其中有一个表存在满足条件的记录，就返回行。
SELF JOIN	将一个表连接到自身，就像该表是两个表一样。为了区分两个表，在 SQL 语句中需要至少重命名一个表。
CROSS JOIN	交叉连接，从两个或者多个连接表中返回记录集的笛卡尔积。

is null#空值判断不用=
SUM(IF(rating < 3, 1, 0))
sum(first=customer_pref_delivery_date)
使用group by 后面几列都是以group by的列索引
CASE
    WHEN x + y > z AND x + z > y AND y + z > x THEN 'Yes'
    ELSE 'No'
END AS 'triangle'
