import json,subprocess,sys
R=open(r"D:\AI-Proj\gemini-proxy\_t.py",encoding="utf-8").read()
i=R.find('T="')
exec(R[:i])
