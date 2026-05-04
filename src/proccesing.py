def reflect_index(index: int, size: int) -> int:
    if index < 0:
        return -index
    if index >= size:
        return 2 * (size - 1) - index
    return index


def zero_padding():
    pass


def replicate():
    pass


def wrap():
    pass
