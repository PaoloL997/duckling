from duckling.files.pdf.convert import PDF


def warmup():
    conv = PDF()
    res = conv.load(path="2408_09869v5.pdf")
    print(res)


if __name__ == "__main__":
    warmup()
