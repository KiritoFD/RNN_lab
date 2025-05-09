import nbformat as nbf
import re

def convert_py_to_ipynb(py_file_path, ipynb_file_path):
    """
    将Python脚本转换为Jupyter Notebook格式
    参数:
        py_file_path: Python文件的路径
        ipynb_file_path: 要创建的Jupyter Notebook文件的路径
    """
    # 读取Python文件内容
    with open(py_file_path, 'r', encoding='utf-8') as f:
        py_content = f.read()
    
    # 创建一个新的notebook
    nb = nbf.v4.new_notebook()
    
    # 添加标题单元格
    nb.cells.append(nbf.v4.new_markdown_cell("# RNN/LSTM 中文影评情感分类实验"))
    
    # 分割主要部分
    parts = re.split(r'# ={10,}\n# (.*?)\n# ={10,}', py_content)
    
    if len(parts) > 1:
        # 第一部分是导言和设置
        intro_code = parts[0].strip()
        if intro_code:
            nb.cells.append(nbf.v4.new_code_cell(intro_code))
        
        # 处理其余部分
        for i in range(1, len(parts), 2):
            if i < len(parts):
                # 添加标题作为markdown单元格
                section_title = parts[i].strip()
                nb.cells.append(nbf.v4.new_markdown_cell(f"## {section_title}"))
                
                if i+1 < len(parts):
                    section_content = parts[i+1].strip()
                    
                    # 检查是否原来是Markdown单元格
                    if "Markdown单元格" in section_title:
                        # 将注释转换为Markdown文本
                        markdown_content = section_content.replace('# ', '')
                        nb.cells.append(nbf.v4.new_markdown_cell(markdown_content))
                    else:
                        # 将代码按空行分割成多个单元格
                        code_blocks = re.split(r'\n\s*\n', section_content)
                        for block in code_blocks:
                            if block.strip():
                                # 检查是否有特殊的注释模式表示这是一个Markdown块
                                if block.strip().startswith('# 结论与改进方向'):
                                    markdown_text = block.strip().replace('# ', '')
                                    nb.cells.append(nbf.v4.new_markdown_cell(markdown_text))
                                else:
                                    nb.cells.append(nbf.v4.new_code_cell(block.strip()))
    else:
        # 如果没有找到主要部分分隔符，则将整个内容作为代码单元格
        chunks = re.split(r'\n\s*\n', py_content)
        for chunk in chunks:
            if chunk.strip():
                nb.cells.append(nbf.v4.new_code_cell(chunk.strip()))
    
    # 保存notebook
    with open(ipynb_file_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    convert_py_to_ipynb("c:\\GitHub\\RNN_lab\\rnn_tutorial.py", "c:\\GitHub\\RNN_lab\\RNN.ipynb")
    print("转换完成！生成的notebook文件为：c:\\GitHub\\RNN_lab\\RNN.ipynb")
