from index_manager import rebuild_index


def build_index():
    return rebuild_index(reason="build_index.py compatibility entrypoint")


if __name__ == "__main__":
    build_index()
