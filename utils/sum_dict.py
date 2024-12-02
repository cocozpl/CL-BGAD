
def dict_addto(a, b):
    res = {}
    for k in b:
        if k in a:
            res[k] = a[k] + float(b[k])
        else:
            res[k] = float(b[k])
    return res


def dict_div(res: dict, div) -> dict:
    for k in res.keys():
        res[k] /= div
    return res
