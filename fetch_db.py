import pymysql
from dotenv import load_dotenv
import os

def get_print_data(num_variations: int, num_datas_each: int, is_front: bool):
    # .env에서 커넥션 정보 불러오기
    load_dotenv()
    host = os.environ.get('Host')
    port = os.environ.get('Port')
    user = os.environ.get('User')
    password = os.environ.get('Password')
    db = os.environ.get('DB')

    substr = 'not defined yet'

    # mySql과의 커넥션 정의
    conn = pymysql.connect(host=host, port=int(
        port), user=user, password=password, db=db, charset='utf8mb4')

    # 커넥션의 커서를 정의
    cursor = conn.cursor()

    if is_front:
        substr = 'print_front'
    else:
        substr = 'print_back'

    # print의 종류를 가져옴
    cursor.execute(f"""
        select distinct {substr}
        from label_data l, image_data i
        where l.image_name = i.file_name
        limit {num_variations};
    """)
    # variations = [["print1",], ["print2",], ...]
    variations = cursor.fetchall()
    print(variations)

    datas = []

    if is_front:
        substr = '앞면'
    else:
        substr = '뒷면'

    # 각 variations에 대해 10개씩 이미지를 fetch
    for [i] in variations:
        print(i)
        cursor.execute(f"""
            select i.image, l.print_front
            from label_data l, image_data i
            where l.image_name IN (
                select file_name
                from image_data i
            ) and l.image_name = i.file_name 
            and l.print_front = '{i}'
            and l.drug_dir = '{substr}'
            order by rand()
            limit {num_datas_each};
        """)
        res = cursor.fetchall()
        for j in res:
            datas.append(j)

    train_data_x = []
    train_data_y = []
    test_data_x = []
    test_data_y = []

    start_index = 0
    end_index = len(datas) - 1
    step = num_datas_each
    r = step * len(variations)

    for i in range(r):
        if (i - start_index) % step == 0:
            test_data_x.append((datas[i][0]))
            test_data_y.append((datas[i][1]))
        else:
            train_data_x.append((datas[i][0]))
            train_data_y.append((datas[i][1]))

    return (train_data_x, train_data_y), (test_data_x, test_data_y)