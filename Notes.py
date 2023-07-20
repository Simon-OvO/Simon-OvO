#download dataset into browser 
from pyodide.http import pyfetch

#使用异步函数（async）调取数据可以避免长时等待数据调取
#使用await表示等待异步函数完成后再执行下一步操作
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
}
