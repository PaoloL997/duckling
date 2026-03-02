from duckling.graph import DucklingGraph


def warmup():
    graph = DucklingGraph()
    state = graph.run(path="2408_09869v5.pdf")
    print(state)


if __name__ == "__main__":
    warmup()
