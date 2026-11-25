import httpx,json
H={"Authorization":"Bearer rnd_8El1E5WO8lA86Rbnc32bQSE5pnuP"}
o=httpx.get("https://api.render.com/v1/owners",headers=H).json()
print(o[0]["owner"]["id"])
