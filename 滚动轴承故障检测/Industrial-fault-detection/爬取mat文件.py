import requests
import time
def spider(num):
    # 定义要下载的文件URL
    url = f'https://engineering.case.edu/sites/default/files/{num}.mat'

    # 使用requests库进行文件下载
    response = requests.get(url)

    # 检查请求是否成功
    if response.status_code == 200:
        # 将文件保存为本地文件
        with open(f'{num}.mat', 'wb') as file:
            file.write(response.content)
        print(f"文件已成功下载并保存为 '{num}.mat'")
    else:
        print(f"文件下载失败，状态码: {response.status_code}")

# for i in range(300):
#     time.sleep(3)
spider(97)