select column_names as names from (select fuction() as name from table_name group by column_name4 )
  where column_name like '%*%' order by column_name5 desc
select date_add(week(now()),interval n week)#日期加减
select datediff(date1,date2)/timediff(time1,time2)#时间间隔
select substr(salary,locate('-',salary)+1,length(salary)-locate('-',salary)-1)#数据清洗获取中间字段
