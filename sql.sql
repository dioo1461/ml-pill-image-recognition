use pilldata;

select distinct print_front
from label_data
limit 30;

select l.print_front, i.file_name, i.image
from label_data l, image_data i
where l.image_name IN (
	select file_name
    from image_data i
) and l.image_name = i.file_name
limit 100;

select l.print_front, count(*)
from label_data l, image_data i
where l.image_name = i.file_name
group by l.print_front;

select i.image, l.print_front
from label_data l, image_data i
where l.image_name IN (
	select file_name
	from image_data i
) and l.image_name = i.file_name and l.print_front = 'SEL'
limit 10;

select l.print_front, l.print_back
from label_data l, image_data i
where l.image_name IN (
	select file_name
	from image_data i
) and l.image_name = i.file_name and l.print_front = 'SEL'
limit 10;

select distinct print_front
    from label_data l, image_data i
    where l.image_name = i.file_name
    limit 30;