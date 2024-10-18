import tkinter as tk
from tkinter import ttk
import speech_recognition as sr
import threading  # 引入线程模块，以避免UI冻结
from PIL import Image, ImageTk
#######################检索界面生成###############################
def predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jieba.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    cat_id = pred.argmax(axis=1)[0]
    res = cat_id_df[cat_id_df.cat_id == cat_id]['cat'].values[0]
    return res

from tkinter import ttk
# 创建窗口
root = tk.Tk()
root.title('意图检索引擎')
root.geometry('810x400+100+100')
root.resizable(False, False) # 禁止调整窗口大小
root.iconbitmap(r'F:\\ZJD\\本科毕设\\桌面数据集、图标\\bitbug_favicon.ico')
# 背景图
img = tk.PhotoImage(file=r'背景图.png')
tk.Label(root, image=img).pack()
# 搜索文本框
search_frame = tk.Frame(root)
search_frame.pack(pady=12)

# 创建一个字符串变量和历史列表
search_va = tk.StringVar()
history_list = []
history_index = -1

# 加载图标
search_icon_path =r'bitbug_favicon.ico'  
search_icon_image = Image.open(search_icon_path)
search_icon_image = search_icon_image.resize((20, 20))  
search_icon_tk = ImageTk.PhotoImage(search_icon_image)


tk.Label(search_frame, text='请输入文本', font=('黑体', 13)).pack(side=tk.LEFT, padx=6)
tk.Entry(search_frame, relief='flat', width=30, textvariable=search_va).pack(side=tk.LEFT, padx=(6,0), fill='both')
tk.Button(search_frame, image=search_icon_tk, font=('黑体', 12), relief='flat', bg='white',height=28,
          command=lambda: update_output(search_va.get())).pack(side=tk.LEFT, padx=(0,5))
search_frame.search_icon_tk = search_icon_tk

# 搜索结果显示框和滚动条
tree_view = ttk.Treeview(root,show="headings", columns=('num','text', 'search results'))
tree_view.column('num', width=5, anchor='center')
tree_view.column('text', width=300, anchor='w')
tree_view.column('search results', width=100, anchor='center')
tree_view.heading('num', text='序号')
tree_view.heading('text', text='输入的文本')
tree_view.heading('search results', text='意图检索结果')
tree_view.pack(fill=tk.BOTH, expand=False, pady=10)

# 添加语音识别功能
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("请说话...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language='zh-CN')
        print("您说的是: " + text)
        search_va.set(text)  # 将识别的文本设置到输入框
        update_output(text)  # 自动触发查询
    except sr.UnknownValueError:
        print("很抱歉无法理解您说的话")
    except sr.RequestError as e:
        print(f"无法从Google Speech Recognition服务请求结果; {e}")

def start_voice_input():
    """启动一个新的线程来执行语音识别，防止UI冻结"""
    thread = threading.Thread(target=recognize_speech)
    thread.start()

# 添加语音输入按钮
tk.Button(search_frame, text='语音输入', font=('黑体', 12), relief='flat', bg='#fe6b00',
          command=start_voice_input).pack(side=tk.LEFT, padx=5)

# 定义更新输出框的函数
def update_output(value):
    global history_list, history_index
    result=predict(str(value))  
    history_list = [result] + history_list[:-1]  
    history_index += 1  # 更新历史记录索引
    # 在 Treeview 中插入数据，序号从1开始
    tree_view.insert("", "end", values=(str(history_index + 1), value, result))
    root.title('意图检索 - ' + str(history_index) + ' / ' + str(len(history_list)))  # 更新窗口标题，显示历史记录数量和总数
    search_va.set('')  # 清空输入框中的内容，以便下一次搜索时输入新的关键词

root.mainloop()
