"""组长端（10-26，Claude）：给某位同学发一条批注（同学的 agent 下次同步时读到）。
用法：python tools/send_note.py 同学甲 "暂停"
      python tools/send_note.py 同学乙 "立即同步"
允许的指令见仓库首页 README 第 4 节；其他文字只会被转告给同学本人。发出的批注同时记在 <同学>/批注记录.md。"""
import datetime
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUFFIX = 'd4a86d'
SLUG = {'同学甲': 'jia', '同学乙': 'yi', '同学丙': 'bing'}

name, text = sys.argv[1], sys.argv[2]
q = urllib.parse.urlencode({'title': f'给{name}的批注'})
req = urllib.request.Request(f'https://ntfy.sh/railseg-{SUFFIX}-notes-{SLUG[name]}?{q}', data=text.encode('utf-8'), method='POST')
r = json.loads(urllib.request.urlopen(req, timeout=60).read())
d = REPO / name; d.mkdir(exist_ok=True)
with open(d / '批注记录.md', 'a', encoding='utf-8', newline='\n') as f:
    f.write(f"\n- {datetime.datetime.now():%Y-%m-%d %H:%M}（编号 {r['id']}）：{text}\n")
print('已发送', r['id'], text)
